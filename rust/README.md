# Vineyard Rust SDK

> [!NOTE]
> Rust nightly is required. The vineyard Rust SDK is still under development.
> The API may change in the future.

[![crates.io](https://img.shields.io/crates/v/vineyard.svg)](https://crates.io/crates/vineyard)
[![Downloads](https://img.shields.io/crates/d/vineyard)](https://crates.io/crates/vineyard)
[![Docs.rs](https://img.shields.io/docsrs/vineyard/latest)](https://docs.rs/vineyard/latest/vineyard/)

Connecting to Vineyard
----------------------

- Resolve the UNIX-domain socket from the environment variable `VINEYARD_IPC_SOCKET`:

    ```rust
    use vineyard::client::*;

    let mut client = vineyard::default().unwrap();
    ```

- Or, using explicit parameter:

    ```rust
    use vineyard::client::*;

    let mut client = vineyard::connect("/var/run/vineyard.sock").unwrap();
    ```

Interact with Vineyard
----------------------

- Creating blob:

    ```rust
    let mut blob_writer = client.create_blob(N)?;
    ```

- Get object:

    ```rust
    let mut meta_writer = client.get::<DataFrame>(object_id)?;
    ```

Inter-op with Python: `numpy.ndarray`
-------------------------------------

- Python:

    ```python
    import numpy as np
    import vineyard

    client = vineyard.connect()

    np_array = np.random.rand(10, 20).astype(np.int32)
    object_id = int(client.put(np_array))
    ```

- Rust:

    ```rust
    let mut client = IPCClient::default()?;
    let tensor = client.get::<Int32Tensor>(object_id)?;
    assert_that!(tensor.shape().to_vec()).is_equal_to(vec![10, 20]);
    ```

Inter-op with Python: `pandas.DataFrame`
----------------------------------------

- Python

    ```python
    import pandas as pd
    import vineyard

    client = vineyard.connect()

    df = pd.DataFrame({'a': ["1", "2", "3", "4"], 'b': ["5", "6", "7", "8"]})
    object_id = int(client.put(df))
    ```

- Rust

    ```rust
    let mut client = IPCClient::default()?;
    let dataframe = client.get::<DataFrame>(object_id)?;
    assert_that!(dataframe.num_columns()).is_equal_to(2);
    assert_that!(dataframe.names().to_vec()).is_equal_to(vec!["a".into(), "b".into()]);
    for index in 0..dataframe.num_columns() {
        let column = dataframe.column(index);
        assert_that!(column.len()).is_equal_to(4);
    }
    ```

Inter-op with Python: `pyarrow.RecordBatch`
-------------------------------------

- Python

    ```python
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
    ```

- Rust

    ```rust
    let batch = client.get::<RecordBatch>(object_id)?;
    assert_that!(batch.num_columns()).is_equal_to(3);
    assert_that!(batch.num_rows()).is_equal_to(4);
    let schema = batch.schema();
    let names = ["f0", "f1", "f2"];
    let recordbatch = batch.as_ref().as_ref();
    ```

Inter-op with Python: `pyarrow.Table`
-------------------------------------

- Python

    ```python
    batches = [batch] * 5
    table = pa.Table.from_batches(batches)
    object_id = int(client.put(table))
    ```

- Rust

    ```rust
    let mut client = IPCClient::default()?;
    let table = client.get::<Table>(object_id)?;
    assert_that!(table.num_batches()).is_equal_to(5);
    for batch in table.batches().iter() {
        // ...
    }
    ```

Inter-op with Python: `polars.DataFrame`
----------------------------------------

- Python

    ```python
    import polars

    dataframe = polars.DataFrame(table)
    object_id = int(client.put(dataframe))
    ```

- Rust

    ```rust
    use vineyard_polars::ds::dataframe::DataFrame;

    let mut client = IPCClient::default()?;
    let dataframe = client.get::<DataFrame>(object_id)?;
    let dataframe = dataframe.as_ref().as_ref();
    assert_that!(dataframe.width()).is_equal_to(3);
    for column in dataframe.get_columns() {
        // ...
    }
    ```

Inter-op with Python: `polars.DataFrame`
----------------------------------------

- Python

    ```python
    batches = [batch] * 5
    table = pa.Table.from_batches(batches)
    object_id = int(client.put(table))
    ```

- Rust

    ```rust
    use vineyard_datafusion::ds::dataframe::DataFrame;

    let mut client = IPCClient::default()?;
    let dataframe = client.get::<DataFrame>(object_id)?;

    let ctx = SessionContext::new();
    let table = ctx.read_table(dataframe.table_provider()).unwrap();

    assert_that!(block_on(table.count()).unwrap()).is_equal_to(1000);
    ```
