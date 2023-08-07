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
    use std::fmt::Debug;
    use std::rc::Rc;

    use num_traits::FromPrimitive;
    use spectral::prelude::*;

    use super::super::super::client::*;
    use super::super::array::*;

    fn test_array_generic<T: TypeName + FromPrimitive + PartialEq + Debug + 'static>() {
        const N: usize = 1024;

        let mut conn = IPCClient::default().unwrap();
        let client = Rc::get_mut(&mut conn).unwrap();

        let mut builder = ArrayBuilder::<T>::new(client, N).unwrap();

        let slice_mut = builder.as_mut_slice();
        for i in 0..N {
            slice_mut[i] = T::from_usize(i).unwrap();
        }

        let array_object_id: ObjectID;
        // test seal
        {
            let object = builder.seal(client).unwrap();
            let array = downcast_object::<Array<T>>(object).unwrap();
            let blob_id = array.id();
            assert_that(&blob_id).is_greater_than(0);

            let slice = array.as_slice();
            for i in 0..N {
                assert_that(&slice[i]).is_equal_to(T::from_usize(i).unwrap());
            }
            array_object_id = array.id();
        }

        // test get array
        {
            let array = client.get::<Array<T>>(array_object_id).unwrap();
            let array_id = array.id();
            assert_that(&array_id).is_greater_than(0);
            assert_that(&array_id).is_equal_to(array_object_id);

            let slice = array.as_slice();
            for i in 0..N {
                assert_that(&slice[i]).is_equal_to(T::from_usize(i).unwrap());
            }
        }
    }

    #[test]
    fn test_array_int32() {
        test_array_generic::<i32>();
    }

    #[test]
    fn test_array_int64() {
        test_array_generic::<i64>();
    }

    #[test]
    fn test_array_float() {
        test_array_generic::<f32>();
    }

    #[test]
    fn test_array_double() {
        test_array_generic::<f64>();
    }
}
