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
    use inline_python::{python, Context};
    use spectral::prelude::*;

    use vineyard::client::*;
    use vineyard::ds::arrow::*;

    #[test]
    fn test_arrow_recordbatch() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import pandas as pd
            import pyarrow as pa
            import vineyard

            client = vineyard.connect()

            arrays = [
                pa.array([1, 2, 3, 4]),
                pa.array(["foo", "bar", "baz", "qux"]),
                pa.array([3.0, 5.0, 7.0, 9.0]),
            ]
            batch = pa.RecordBatch.from_arrays(arrays, ["f0", "f1", "f2"])
            object_id = int(client.put(batch))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let batch = client.get::<RecordBatch>(object_id)?;
        assert_that!(batch.num_columns()).is_equal_to(3);
        assert_that!(batch.num_rows()).is_equal_to(4);
        let schema = batch.schema();
        let names = ["f0", "f1", "f2"];
        let recordbatch = batch.as_ref().as_ref();
        for (index, name) in names.into_iter().enumerate() {
            assert_that!(schema.field(index).name()).is_equal_to(&name.to_string());

            let column = recordbatch.column(index);
            match index {
                0 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .unwrap();
                    let expected: arrow_array::Int64Array = vec![1, 2, 3, 4].into();
                    assert_that!(array).is_equal_to(&expected);
                }
                1 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .unwrap();
                    let expected: arrow_array::LargeStringArray =
                        vec!["foo", "bar", "baz", "qux"].into();
                    assert_that!(array).is_equal_to(&expected);
                }
                2 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .unwrap();
                    let expected: arrow_array::Float64Array = vec![3.0, 5.0, 7.0, 9.0].into();
                    assert_that!(array).is_equal_to(&expected);
                }
                _ => unreachable!(),
            }
        }
        return Ok(());
    }

    #[test]
    fn test_arrow_table() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import pandas as pd
            import pyarrow as pa
            import vineyard
            client = vineyard.connect()

            arrays = [
                pa.array([1, 2, 3, 4]),
                pa.array(["foo", "bar", "baz", "qux"]),
                pa.array([3.0, 5.0, 7.0, 9.0]),
            ]
            batch = pa.RecordBatch.from_arrays(arrays, ["f0", "f1", "f2"])
            batches = [batch] * 5
            table = pa.Table.from_batches(batches)
            object_id = int(client.put(table))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let table = client.get::<Table>(object_id)?;
        assert_that!(table.num_batches()).is_equal_to(5);
        for batch in table.batches().iter() {
            assert_that!(batch.num_columns()).is_equal_to(3);
            assert_that!(batch.num_rows()).is_equal_to(4);
            let schema = batch.schema();
            let names = ["f0", "f1", "f2"];
            let recordbatch = batch.as_ref().as_ref();
            for (index, name) in names.into_iter().enumerate() {
                assert_that!(schema.field(index).name()).is_equal_to(&name.to_string());

                let column = recordbatch.column(index);
                match index {
                    0 => {
                        let array = column
                            .as_any()
                            .downcast_ref::<arrow_array::Int64Array>()
                            .unwrap();
                        let expected: arrow_array::Int64Array = vec![1, 2, 3, 4].into();
                        assert_that!(array).is_equal_to(&expected);
                    }
                    1 => {
                        let array = column
                            .as_any()
                            .downcast_ref::<arrow_array::LargeStringArray>()
                            .unwrap();
                        let expected: arrow_array::LargeStringArray =
                            vec!["foo", "bar", "baz", "qux"].into();
                        assert_that!(array).is_equal_to(&expected);
                    }
                    2 => {
                        let array = column
                            .as_any()
                            .downcast_ref::<arrow_array::Float64Array>()
                            .unwrap();
                        let expected: arrow_array::Float64Array = vec![3.0, 5.0, 7.0, 9.0].into();
                        assert_that!(array).is_equal_to(&expected);
                    }
                    _ => unreachable!(),
                }
            }
        }
        return Ok(());
    }
}
