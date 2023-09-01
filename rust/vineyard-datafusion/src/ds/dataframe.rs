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

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema};
use datafusion::common::DataFusionError;
use datafusion::datasource::memory::MemTable;
use datafusion::datasource::TableProvider;

use vineyard::client::*;
use vineyard::ds::arrow::{Table, TableBuilder};
use vineyard::ds::dataframe::DataFrame as VineyardDataFrame;

/// Convert a datafusion error to a Vineyard error, as orphan impls are not allowed
/// in Rust
///
/// Usage:
///
/// ```ignore
/// let x = ctx.select(...).map_err(error)?;
/// ```
pub fn error(error: DataFusionError) -> VineyardError {
    VineyardError::invalid(format!("{}", error))
}

#[derive(Debug)]
pub struct DataFrame {
    meta: ObjectMeta,
    dataframe: MemTable,
}

impl Default for DataFrame {
    fn default() -> Self {
        DataFrame {
            meta: ObjectMeta::default(),
            dataframe: MemTable::try_new(Arc::new(Schema::new(Vec::<Field>::new())), vec![])
                .unwrap(),
        }
    }
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
        let recordbatch = dataframe.recordbatch()?;
        self.meta = dataframe.metadata();
        self.dataframe =
            MemTable::try_new(recordbatch.schema(), vec![vec![recordbatch]]).map_err(error)?;
        return Ok(());
    }

    fn construct_from_arrow_table(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Table>(), meta.get_typename()?)?;
        let table = downcast_object::<Table>(Table::new_boxed(meta)?)?;
        let schema = table.schema().clone();
        let batches: Vec<Vec<RecordBatch>> = table
            .batches()
            .iter()
            .map(|batch| vec![batch.as_ref().as_ref().clone()])
            .collect();
        self.meta = table.metadata();
        self.dataframe = MemTable::try_new(Arc::new(schema), batches).map_err(error)?;
        return Ok(());
    }

    pub fn table(self) -> MemTable {
        return self.dataframe;
    }

    pub fn table_provider(self) -> Arc<dyn TableProvider> {
        return Arc::new(self.dataframe);
    }
}

impl AsRef<MemTable> for DataFrame {
    fn as_ref(&self) -> &MemTable {
        &self.dataframe
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
    pub fn new(client: &mut IPCClient, batches: &[RecordBatch]) -> Result<Self> {
        assert!(
            !batches.is_empty(),
            "cannot build a dataframe from empty record batch collections"
        );
        return Ok(ArrowDataFrameBuilder(TableBuilder::new(
            client,
            batches[0].schema().as_ref(),
            batches,
        )?));
    }
}
