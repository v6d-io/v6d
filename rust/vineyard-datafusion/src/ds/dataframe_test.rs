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

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use std::sync::Arc;

    use arrow_array as array;
    use arrow_schema as schema;
    use datafusion::prelude::*;
    use spectral::prelude::*;

    use crate::ds::dataframe::{error, ArrowDataFrameBuilder, DataFrame};
    use vineyard::client::*;
    use vineyard::{get, put};

    #[test]
    fn test_polars_dataframe() -> Result<()> {
        const N: usize = 1024;
        let mut client = IPCClient::default()?;

        // prepare data
        let vec0 = (0..N).map(|i| i as i32).collect::<Vec<_>>();
        let vec1 = (0..N).map(|i| i as f64).collect::<Vec<_>>();
        let vec2 = (0..N).map(|i| format!("{:?}", i)).collect::<Vec<_>>();
        let array0 = array::Int32Array::from(vec0);
        let array1 = array::Float64Array::from(vec1);
        let array2 = array::LargeStringArray::from(vec2);

        let schema = schema::Schema::new(vec![
            schema::Field::new("f0", schema::DataType::Int32, false),
            schema::Field::new("f1", schema::DataType::Float64, false),
            schema::Field::new("f2", schema::DataType::LargeUtf8, false),
        ]);
        let batch = array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(array0), Arc::new(array1), Arc::new(array2)],
        )?;
        let batches = vec![batch];

        let object = put!(&mut client, ArrowDataFrameBuilder, &batches)?;
        let object_id = object.id();
        let dataframe = get!(client, DataFrame, object_id)?;

        // test sql on dataframe
        let ctx = SessionContext::new();
        let table = ctx.read_table(dataframe.table_provider()).map_err(error)?;

        assert_that!(block_on(table.count()).map_err(error)?).is_equal_to(N);
        return Ok(());
    }
}
