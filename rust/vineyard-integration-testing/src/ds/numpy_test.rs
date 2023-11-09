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
    use vineyard::ds::tensor::{Float64Tensor, Int32Tensor, StringTensor};

    #[test]
    fn test_numpy_int32() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import numpy as np
            import vineyard

            client = vineyard.connect()

            np_array = np.random.rand(10, 20).astype(np.int32)
            object_id = int(client.put(np_array))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let tensor = client.get::<Int32Tensor>(object_id)?;
        assert_that!(tensor.shape().to_vec()).is_equal_to(vec![10, 20]);
        return Ok(());
    }

    #[test]
    fn test_numpy_float64() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import numpy as np
            import vineyard

            client = vineyard.connect()

            np_array = np.random.rand(10, 20)
            object_id = int(client.put(np_array))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let tensor = client.get::<Float64Tensor>(object_id)?;
        assert_that!(tensor.shape().to_vec()).is_equal_to(vec![10, 20]);
        return Ok(());
    }

    #[test]
    fn test_numpy_string() -> Result<()> {
        use arrow_array::array::Array;

        let ctx = Context::new();
        ctx.run(python! {
            import numpy as np
            import vineyard

            client = vineyard.connect()

            np_array = np.array(['a', 'b', 'c', 'd', 'e'])
            object_id = int(client.put(np_array))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let tensor = client.get::<StringTensor>(object_id)?;
        assert_that!(tensor.shape().to_vec()).is_equal_to(vec![5]);
        let array = tensor.as_ref().as_ref();
        assert_that!(array.len()).is_equal_to(5);
        for index in 0..array.len() {
            assert_that!(array.value(index).to_string())
                .is_equal_to(format!("{}", (index as u8 + b'a') as char));
        }
        return Ok(());
    }
}
