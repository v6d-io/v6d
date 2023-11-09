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
    use vineyard::ds::dataframe::DataFrame;

    #[test]
    fn test_pandas_int() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import numpy as np
            import pandas as pd
            import vineyard

            client = vineyard.connect()

            df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
            object_id = int(client.put(df))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let dataframe = client.get::<DataFrame>(object_id)?;
        assert_that!(dataframe.num_columns()).is_equal_to(2);
        assert_that!(dataframe.names().to_vec()).is_equal_to(vec!["a".into(), "b".into()]);
        for index in 0..dataframe.num_columns() {
            let column = dataframe.column(index);
            assert_that!(column.len()).is_equal_to(4);
        }
        return Ok(());
    }

    #[test]
    fn test_pandas_string() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import numpy as np
            import pandas as pd
            import vineyard

            client = vineyard.connect()

            df = pd.DataFrame({'a': ["1", "2", "3", "4"], 'b': ["5", "6", "7", "8"]})
            object_id = int(client.put(df))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let dataframe = client.get::<DataFrame>(object_id)?;
        assert_that!(dataframe.num_columns()).is_equal_to(2);
        assert_that!(dataframe.names().to_vec()).is_equal_to(vec!["a".into(), "b".into()]);
        for index in 0..dataframe.num_columns() {
            let column = dataframe.column(index);
            assert_that!(column.len()).is_equal_to(4);
        }
        return Ok(());
    }
}
