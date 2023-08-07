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
    use std::rc::Rc;

    use spectral::prelude::*;

    use super::super::super::*;
    use super::super::*;

    #[test]
    fn test_blob() {
        const N: usize = 1024;

        let mut conn = IPCClient::default().unwrap();
        let client = Rc::get_mut(&mut conn).unwrap();

        let blob_writer = client.create_blob(N).unwrap();
        let blob_writer_id = blob_writer.id();
        assert_that(&blob_writer_id).is_greater_than(0);

        let slice_mut = blob_writer.as_mut_slice();
        for i in 0..N {
            slice_mut[i] = i as u8;
        }

        // test seal
        {
            let object = blob_writer.seal(client).unwrap();
            let blob = downcast_object::<Blob>(object).unwrap();
            let blob_id = blob.id();
            assert_that(&blob_id).is_greater_than(0);
            assert_that(&blob_id).is_equal_to(blob_writer_id);

            let slice = blob.as_slice().unwrap();
            for i in 0..N {
                assert_that(&slice[i]).is_equal_to(i as u8);
            }
        }

        // test get blob
        {
            let blob = client.get_blob(blob_writer_id).unwrap();
            let blob_id = blob.id();
            assert_that(&blob_id).is_greater_than(0);
            assert_that(&blob_id).is_equal_to(blob_writer_id);

            let slice = blob.as_slice().unwrap();
            for i in 0..N {
                assert_that(&slice[i]).is_equal_to(i as u8);
            }
        }
    }
}
