// Copyright 2020-2023 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow2::array;
use arrow2::datatypes;
use itertools::izip;
use polars_core::prelude as polars;
use serde_json::Value;

use vineyard::client::*;
use vineyard::ds::arrow::{Table, TableBuilder};
use vineyard::ds::dataframe::DataFrame as VineyardDataFrame;

/// Convert a Polars error to a Vineyard error, as orphan impls are not allowed
/// in Rust
///
/// Usage:
///
/// ```ignore
/// let x = polars::DataFrame::new(...).map_err(error)?;
/// ```
pub fn error(error: polars::PolarsError) -> VineyardError {
    VineyardError::invalid(format!("{}", error))
}

#[derive(Debug, Default)]
pub struct DataFrame {
    meta: ObjectMeta,
    dataframe: polars::DataFrame,
}

impl_typename!(DataFrame, "vineyard::Table");

impl Object for DataFrame {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        let ty = meta.get_typename()?;
        if ty == typename::<VineyardDataFrame>() {
            return self.construct_from_pandas_dataframe(meta);
        } else if ty == typename::<Table>() {
            return self.construct_from_arrow_table(meta);
        } else {
            return Err(VineyardError::type_error(format!(
                "cannot construct DataFrame from this metadata: {}",
                ty
            )));
        }
    }
}

register_vineyard_object!(DataFrame);

impl DataFrame {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut object = Box::<Self>::default();
        object.construct(meta)?;
        Ok(object)
    }

    fn construct_from_pandas_dataframe(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<VineyardDataFrame>(), meta.get_typename()?)?;
        let dataframe = downcast_object::<VineyardDataFrame>(VineyardDataFrame::new_boxed(meta)?)?;
        let names = dataframe.names().to_vec();
        let columns: Vec<Box<dyn array::Array>> = dataframe
            .columns()
            .iter()
            .map(|c| array::from_data(&c.array().to_data()))
            .collect();
        let series: Vec<polars_core::series::Series> = names
            .iter()
            .zip(columns)
            .map(|(name, column)| {
                let datatype = polars::DataType::from(column.data_type());
                unsafe {
                    polars_core::series::Series::from_chunks_and_dtype_unchecked(
                        name,
                        vec![column],
                        &datatype,
                    )
                }
            })
            .collect::<Vec<_>>();
        self.meta = dataframe.metadata();
        self.dataframe = polars::DataFrame::new(series).map_err(error)?;
        return Ok(());
    }

    fn construct_from_arrow_table(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Table>(), meta.get_typename()?)?;
        let table = downcast_object::<Table>(Table::new_boxed(meta)?)?;
        let schema = table.schema();
        let names = schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        let types = schema
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>();
        let mut columns: Vec<Vec<Box<dyn array::Array>>> = Vec::with_capacity(table.num_columns());
        for index in 0..table.num_columns() {
            let mut chunks = Vec::with_capacity(table.num_batches());
            for batch in table.batches() {
                let batch = batch.as_ref().as_ref();
                let chunk = batch.column(index);
                chunks.push(array::from_data(&chunk.to_data()));
            }
            columns.push(chunks);
        }
        let series: Vec<polars_core::series::Series> = izip!(&names, types, columns)
            .map(|(name, datatype, chunks)| unsafe {
                polars_core::series::Series::from_chunks_and_dtype_unchecked(
                    name,
                    chunks,
                    &polars::DataType::from(&datatypes::DataType::from(datatype)),
                )
            })
            .collect::<Vec<_>>();
        self.meta = table.metadata();
        self.dataframe = polars::DataFrame::new(series).map_err(error)?;
        return Ok(());
    }
}

impl AsRef<polars::DataFrame> for DataFrame {
    fn as_ref(&self) -> &polars::DataFrame {
        &self.dataframe
    }
}

/// Building a polars dataframe into a pandas-compatible dataframe.
pub struct PandasDataFrameBuilder {
    sealed: bool,
    names: Vec<String>,
    columns: Vec<Box<dyn Object>>,
}

impl ObjectBuilder for PandasDataFrameBuilder {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl ObjectBase for PandasDataFrameBuilder {
    fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<DataFrame>());
        meta.add_usize("__values_-size", self.names.len());
        meta.add_isize("partition_index_row_", -1);
        meta.add_isize("partition_index_column_", -1);
        meta.add_isize("row_batch_index_", -1);
        for (index, (name, column)) in self.names.iter().zip(self.columns).enumerate() {
            meta.add_value(
                &format!("__values_-key-{}", index),
                Value::String(name.into()),
            );
            meta.add_member(&format!("__values_-value-{}", index), column)?;
        }
        let metadata = client.create_metadata(&meta)?;
        return DataFrame::new_boxed(metadata);
    }
}

impl PandasDataFrameBuilder {
    pub fn new(client: &mut IPCClient, dataframe: &polars::DataFrame) -> Result<Self> {
        let mut names = Vec::with_capacity(dataframe.width());
        let mut columns = Vec::with_capacity(dataframe.width());
        for column in dataframe.get_columns() {
            let column = column.rechunk(); // FIXME(avoid copying)
            names.push(column.name().into());
            columns.push(column.chunks()[0].clone());
        }
        return Self::new_from_arrays(client, names, columns);
    }

    pub fn new_from_columns(names: Vec<String>, columns: Vec<Box<dyn Object>>) -> Result<Self> {
        return Ok(PandasDataFrameBuilder {
            sealed: false,
            names,
            columns,
        });
    }

    pub fn new_from_arrays(
        client: &mut IPCClient,
        names: Vec<String>,
        arrays: Vec<Box<dyn array::Array>>,
    ) -> Result<Self> {
        use vineyard::ds::tensor::build_tensor;

        let mut columns = Vec::with_capacity(arrays.len());
        for array in arrays {
            columns.push(build_tensor(client, array.into())?);
        }
        return Ok(PandasDataFrameBuilder {
            sealed: false,
            names,
            columns,
        });
    }
}

/// Building a polars dataframe into a arrow's table-compatible dataframe.
pub struct ArrowDataFrameBuilder(pub TableBuilder);

impl ObjectBuilder for ArrowDataFrameBuilder {
    fn sealed(&self) -> bool {
        self.0.sealed()
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.0.set_sealed(sealed)
    }
}

impl ObjectBase for ArrowDataFrameBuilder {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        self.0.build(client)
    }

    fn seal(self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        let table = downcast_object::<Table>(self.0.seal(client)?)?;
        return DataFrame::new_boxed(table.metadata());
    }
}

impl ArrowDataFrameBuilder {
    pub fn new(client: &mut IPCClient, dataframe: &polars::DataFrame) -> Result<Self> {
        let mut names = Vec::with_capacity(dataframe.width());
        let mut datatypes = Vec::with_capacity(dataframe.width());
        let mut columns = Vec::with_capacity(dataframe.width());
        for column in dataframe.get_columns() {
            names.push(column.name().into());
            datatypes.push(column.dtype().to_arrow());
            columns.push(column.chunks().clone());
        }
        return Self::new_from_columns(client, names, datatypes, columns);
    }

    /// batches[0]: the first record batch
    /// batches[0][0]: the first column of the first record batch
    pub fn new_from_batch_columns(
        client: &mut IPCClient,
        names: Vec<String>,
        datatypes: Vec<datatypes::DataType>,
        num_rows: Vec<usize>,
        num_columns: usize,
        batches: Vec<Vec<Box<dyn Object>>>,
    ) -> Result<Self> {
        let schema = arrow_schema::Schema::new(
            izip!(names, datatypes)
                .map(|(name, datatype)| {
                    arrow_schema::Field::from(datatypes::Field::new(name, datatype, false))
                })
                .collect::<Vec<_>>(),
        );
        return Ok(ArrowDataFrameBuilder(TableBuilder::new_from_batch_columns(
            client,
            &schema,
            num_rows,
            num_columns,
            batches,
        )?));
    }

    /// batches[0]: the first record batch
    /// batches[0][0]: the first column of the first record batch
    pub fn new_from_batches(
        client: &mut IPCClient,
        names: Vec<String>,
        datatypes: Vec<datatypes::DataType>,
        batches: Vec<Vec<Box<dyn array::Array>>>,
    ) -> Result<Self> {
        use vineyard::ds::arrow::build_array;

        let mut num_rows = Vec::with_capacity(batches.len());
        let mut num_columns = 0;
        let mut chunks = Vec::with_capacity(batches.len());
        for batch in batches {
            let mut columns = Vec::with_capacity(batch.len());
            num_columns = columns.len();
            if num_columns == 0 {
                num_rows.push(0);
            } else {
                num_rows.push(batch[0].len());
            }
            for array in batch {
                columns.push(build_array(client, array.into())?);
            }
            chunks.push(columns);
        }
        return Self::new_from_batch_columns(
            client,
            names,
            datatypes,
            num_rows,
            num_columns,
            chunks,
        );
    }

    /// columns[0]: the first column
    /// columns[0][0]: the first chunk of the first column
    pub fn new_from_columns(
        client: &mut IPCClient,
        names: Vec<String>,
        datatypes: Vec<datatypes::DataType>,
        columns: Vec<Vec<Box<dyn array::Array>>>,
    ) -> Result<Self> {
        use vineyard::ds::arrow::build_array;

        let mut num_rows = Vec::new();
        let num_columns = columns.len();
        let mut chunks = Vec::new();
        for (column_index, column) in columns.into_iter().enumerate() {
            for (chunk_index, chunk) in column.into_iter().enumerate() {
                if column_index == 0 {
                    chunks.push(Vec::new());
                    num_rows.push(chunk.len());
                }
                chunks[chunk_index].push(build_array(client, chunk.into())?);
            }
        }
        return Self::new_from_batch_columns(
            client,
            names,
            datatypes,
            num_rows,
            num_columns,
            chunks,
        );
    }
}
