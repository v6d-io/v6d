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
    use polars_core::prelude::NamedFrom;
    use polars_core::series::Series;
    use spectral::prelude::*;

    use vineyard::client::*;
    use vineyard_polars::ds::dataframe::DataFrame;

    #[test]
    fn test_polars_dataframe() -> Result<()> {
        let ctx = Context::new();
        ctx.run(python! {
            import pandas as pd
            import pyarrow as pa
            import polars

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
            dataframe = polars.DataFrame(table)
            object_id = int(client.put(dataframe))
        });
        let object_id = ctx.get::<ObjectID>("object_id");

        let mut client = IPCClient::default()?;
        let dataframe = client.get::<DataFrame>(object_id)?;
        let dataframe = dataframe.as_ref().as_ref();
        assert_that!(dataframe.width()).is_equal_to(3);
        let mut names = Vec::with_capacity(dataframe.width());
        for column in dataframe.get_columns() {
            names.push(column.name());
        }
        assert_that!(names).is_equal_to(vec!["f0", "f1", "f2"]);

        // check column values
        assert_that!(dataframe.column("f0").unwrap().head(Some(4)))
            .is_equal_to(&Series::new("f0", [1, 2, 3, 4]));
        assert_that!(dataframe.column("f1").unwrap().head(Some(4)))
            .is_equal_to(&Series::new("f1", ["foo", "bar", "baz", "qux"]));
        assert_that!(dataframe.column("f2").unwrap().head(Some(4)))
            .is_equal_to(&Series::new("f2", [3.0, 5.0, 7.0, 9.0]));
        return Ok(());
    }
}
