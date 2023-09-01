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

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use itertools::izip;
use serde_json::Value;

use super::tensor::*;
use crate::client::*;

#[derive(Default)]
pub struct DataFrame {
    meta: ObjectMeta,
    names: Vec<String>,
    columns: Vec<Box<dyn Tensor>>,
}

impl_typename!(DataFrame, "vineyard::DataFrame");

impl Object for DataFrame {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        let size = meta.get_usize("__values_-size")?;
        self.names = Vec::with_capacity(size);
        self.columns = Vec::with_capacity(size);
        for i in 0..size {
            let name = meta.get_value(&format!("__values_-key-{}", i))?;
            let name = match name {
                Value::String(name) => name,
                _ => name.to_string(),
            };
            self.names.push(name);
            let column = meta.get_member_untyped(&format!("__values_-value-{}", i))?;
            self.columns.push(downcast_to_tensor(column)?);
        }
        return Ok(());
    }
}

register_vineyard_object!(DataFrame);

impl DataFrame {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut object = Box::<Self>::default();
        object.construct(meta)?;
        Ok(object)
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    pub fn columns(&self) -> &[Box<dyn Tensor>] {
        &self.columns
    }

    pub fn column(&self, index: usize) -> ArrayRef {
        self.columns[index].array()
    }

    pub fn recordbatch(&self) -> Result<RecordBatch> {
        let mut columns = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            columns.push(column.array());
        }
        let types: Vec<DataType> = columns
            .iter()
            .map(|column| column.data_type().clone())
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(
                izip!(self.names.clone(), types)
                    .map(|(name, datatype)| Field::new(name, datatype, true))
                    .collect::<Vec<Field>>(),
            )),
            columns,
        )?;
        return Ok(batch);
    }
}

pub struct DataFrameBuilder {
    sealed: bool,
    names: Vec<String>,
    columns: Vec<Box<dyn Object>>,
}

impl ObjectBuilder for DataFrameBuilder {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl ObjectBase for DataFrameBuilder {
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

impl DataFrameBuilder {
    pub fn new(
        client: &mut IPCClient,
        names: Vec<String>,
        arrays: Vec<arrow_array::ArrayRef>,
    ) -> Result<Self> {
        let mut columns = Vec::with_capacity(arrays.len());
        for array in arrays {
            columns.push(build_tensor(client, array)?);
        }
        return Ok(DataFrameBuilder {
            sealed: false,
            names,
            columns: columns,
        });
    }

    pub fn new_from_columns(names: Vec<String>, columns: Vec<Box<dyn Object>>) -> Result<Self> {
        return Ok(DataFrameBuilder {
            sealed: false,
            names,
            columns,
        });
    }
}
