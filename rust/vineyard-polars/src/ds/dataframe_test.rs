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
    use polars_core::prelude::*;
    use polars_core::series::Series;
    use spectral::prelude::*;

    use crate::ds::dataframe::ArrowDataFrameBuilder;
    use crate::ds::dataframe::DataFrame;
    use vineyard::client::*;
    use vineyard::{get, put};

    #[test]
    fn test_polars_dataframe() -> Result<()> {
        let mut client = IPCClient::default()?;

        let df = df!(
            "f0" => &[1, 2, 3, 4],
            "f1" => &["foo", "bar", "baz", "qux"],
            "f2" => &[3.0, 5.0, 7.0, 9.0],
        )
        .unwrap();

        let object = put!(&mut client, ArrowDataFrameBuilder, &df)?;
        let object_id = object.id();
        let dataframe = get!(client, DataFrame, object_id)?;

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
