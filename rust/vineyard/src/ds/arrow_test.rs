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
    use std::sync::Arc;

    use arrow_array as array;
    use arrow_schema as schema;
    use spectral::prelude::*;

    use super::super::arrow::*;
    use crate::client::*;

    #[test]
    fn test_int_array() -> Result<()> {
        const N: usize = 1024;
        let mut client = IPCClient::default()?;

        // prepare data
        let vec = (0..N).map(|i| i as i32).collect::<Vec<_>>();
        let array = array::Int32Array::from(vec);

        let array_object_id: ObjectID;

        // build into vineyard
        {
            let builder = Int32Builder::new(&mut client, &array)?;
            let object = builder.seal(&mut client)?;
            let array = downcast_object::<Int32Array>(object)?;
            array_object_id = array.id();
            assert_that!(array.len()).is_equal_to(N);

            let slice = array.as_slice();
            for (idx, item) in slice.iter().enumerate() {
                assert_that!(*item).is_equal_to(idx as i32);
            }
        }

        // get from vineyard
        {
            let array = client.get::<Int32Array>(array_object_id).unwrap();
            let array_id = array.id();
            assert_that!(array_id).is_greater_than(0);
            assert_that!(array_id).is_equal_to(array_object_id);
            assert_that!(array.len()).is_equal_to(N);

            let slice = array.as_slice();
            for (idx, item) in slice.iter().enumerate() {
                assert_that!(*item).is_equal_to(idx as i32);
            }
        }

        return Ok(());
    }

    #[test]
    fn test_double_array() -> Result<()> {
        const N: usize = 1024;
        let mut client = IPCClient::default()?;

        // prepare data
        let vec = (0..N).map(|i| i as f64).collect::<Vec<_>>();
        let array = array::Float64Array::from(vec);

        let array_object_id: ObjectID;

        // build into vineyard
        {
            let builder = Float64Builder::new(&mut client, &array)?;
            let object = builder.seal(&mut client)?;
            let array = downcast_object::<Float64Array>(object)?;
            array_object_id = array.id();
            assert_that!(array.len()).is_equal_to(N);

            let slice = array.as_slice();
            for (idx, item) in slice.iter().enumerate() {
                assert_that!(*item).is_equal_to(idx as f64);
            }
        }

        // get from vineyard
        {
            let array = client.get::<Float64Array>(array_object_id).unwrap();
            let array_id = array.id();
            assert_that!(array_id).is_greater_than(0);
            assert_that!(array_id).is_equal_to(array_object_id);
            assert_that!(array.len()).is_equal_to(N);

            let slice = array.as_slice();
            for (idx, item) in slice.iter().enumerate() {
                assert_that!(*item).is_equal_to(idx as f64);
            }
        }

        return Ok(());
    }

    #[test]
    fn test_string_array() -> Result<()> {
        const N: usize = 1024;
        let mut client = IPCClient::default()?;

        // prepare data
        let vec = (0..N).map(|i| format!("{:?}", i)).collect::<Vec<_>>();
        let strings = vec.join("");
        let array = array::LargeStringArray::from(vec);
        let array_object_id: ObjectID;

        // build into vineyard
        {
            let builder = LargeStringBuilder::new(&mut client, &array)?;
            let object = builder.seal(&mut client)?;

            let array = downcast_object::<LargeStringArray>(object)?;
            array_object_id = array.id();
            assert_that!(array.len()).is_equal_to(N);

            let slice = array.as_slice();
            for (item, expect) in slice.iter().zip(strings.as_bytes().iter()) {
                assert_that!(item).is_equal_to(expect);
            }
        }

        // get from vineyard
        {
            let array = client.get::<LargeStringArray>(array_object_id).unwrap();
            let array_id = array.id();
            assert_that!(array_id).is_greater_than(0);
            assert_that!(array_id).is_equal_to(array_object_id);
            assert_that!(array.len()).is_equal_to(N);

            let slice = array.as_slice();
            for (item, expect) in slice.iter().zip(strings.as_bytes().iter()) {
                assert_that!(item).is_equal_to(expect);
            }
        }

        return Ok(());
    }

    #[test]
    fn test_record_batch() -> Result<()> {
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

        let recordbatch_object_id: ObjectID;

        // build into vineyard
        {
            let builder = RecordBatchBuilder::new(&mut client, &batch)?;
            let object = builder.seal(&mut client)?;
            let recordbatch = downcast_object::<RecordBatch>(object)?;
            recordbatch_object_id = recordbatch.id();
            assert_that!(recordbatch.num_rows()).is_equal_to(N);
            assert_that!(recordbatch.num_columns()).is_equal_to(3);

            let recordbatch = recordbatch.as_ref().as_ref();

            let column0 = recordbatch
                .column(0)
                .as_any()
                .downcast_ref::<array::Int32Array>()
                .ok_or(VineyardError::type_error("downcast to Int32Array failed"))?;
            for (idx, item) in column0.iter().enumerate() {
                assert_that!(item).is_equal_to(Some(idx as i32));
            }

            let column1 = recordbatch
                .column(1)
                .as_any()
                .downcast_ref::<array::Float64Array>()
                .ok_or(VineyardError::type_error("downcast to Float64Array failed"))?;
            for (idx, item) in column1.iter().enumerate() {
                assert_that!(item).is_equal_to(Some(idx as f64));
            }

            let column2 = recordbatch
                .column(2)
                .as_any()
                .downcast_ref::<array::LargeStringArray>()
                .ok_or(VineyardError::type_error(
                    "downcast to LargeStringArray failed",
                ))?;
            for (idx, item) in column2.iter().enumerate() {
                assert_that!(item).is_equal_to(Some(format!("{}", idx).as_str()));
            }
        }

        // get from vineyard
        {
            let recordbatch = client.get::<RecordBatch>(recordbatch_object_id).unwrap();
            let recordbatch_id = recordbatch.id();
            assert_that!(recordbatch_id).is_greater_than(0);
            assert_that!(recordbatch_id).is_equal_to(recordbatch_object_id);
            assert_that!(recordbatch.num_rows()).is_equal_to(N);
            assert_that!(recordbatch.num_columns()).is_equal_to(3);

            let recordbatch = recordbatch.as_ref().as_ref();

            let column0 = recordbatch
                .column(0)
                .as_any()
                .downcast_ref::<array::Int32Array>()
                .ok_or(VineyardError::type_error("downcast to Int32Array failed"))?;
            for (idx, item) in column0.iter().enumerate() {
                assert_that!(item).is_equal_to(Some(idx as i32));
            }

            let column1 = recordbatch
                .column(1)
                .as_any()
                .downcast_ref::<array::Float64Array>()
                .ok_or(VineyardError::type_error("downcast to Float64Array failed"))?;
            for (idx, item) in column1.iter().enumerate() {
                assert_that!(item).is_equal_to(Some(idx as f64));
            }

            let column2 = recordbatch
                .column(2)
                .as_any()
                .downcast_ref::<array::LargeStringArray>()
                .ok_or(VineyardError::type_error(
                    "downcast to LargeStringArray failed",
                ))?;
            for (idx, item) in column2.iter().enumerate() {
                assert_that!(item).is_equal_to(Some(format!("{}", idx).as_str()));
            }
        }
        return Ok(());
    }
}
