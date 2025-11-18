Machine Learning with Vineyard
==============================


**Vineyard-ML**: A Vineyard package that integrates Machine Learning Frameworks to Vineyard.

TensorFlow
----------

Using Numpy Data
^^^^^^^^^^^^^^^^

.. code:: python

     >>> import tensorflow as tf
     >>> from vineyard.contrib.ml import tensorflow
     >>> dataset = tf.data.Dataset.from_tensor_slices((data, label))
     >>> data_id = vineyard_client.put(dataset)
     >>> vin_data = vineyard_client.get(data_id)
     
Vineyard supports the ``tf.data.Dataset``. The ``vin_data`` will be a shared-memory object 
from the vineyard.  

Using Dataframe
^^^^^^^^^^^^^^^

.. code:: python

     >>> import pandas as pd
     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'target': [1.0, 2.0, 3.0, 4.0]})
     >>> label = df.pop('target') 
     >>> dataset = tf.data.Dataset.from_tensor_slices((dict(df), label))
     >>> data_id = vineyard_client.put(dataset)
     >>> vin_data = vineyard_client.get(data_id)

Wrap the dataframe with ``tf.data.Dataset``. This enables the use of feature columns as a 
bridge to map from the columns in Pandas Dataframe to features in Dataset. The 
dataset should return a dictionary of column names (from the dataframe) that maps 
to column values. The dataset should only contain ``numerical data``. 

Using RecordBatch of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: python

     >>> import pyarrow as pa
     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'label'])
     >>> data_id = vineyard_client.put(batch)
     >>> vin_data = vineyard_client.get(data_id)

Vineyard supports direct integration of RecordBatch. The ``vin_data`` object will 
be a TensorFlow Dataset, i.e. ``tf.data.Dataset``. Here the ``label`` row should be named as ``label``.

Using Tables of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'label'])
     >>> batches = [batch]*3
     >>> table = pa.Table.from_batches(batches)
     >>> data_id = vineyard_client.put(table)
     >>> vin_data = vineyard_client.get(data_id)

Vineyard supports direct integration of Tables as well. Here, the ``vin_data`` 
object will be of type TensorFlow Dataset, i.e. ``tf.data.Dataset``. Here the ``label`` row 
should be named as ``label``.


PyTorch
-------

Using Numpy Data
^^^^^^^^^^^^^^^^

Vineyard supports ``Custom Datasets`` inherited from the PyTorch Dataset.

.. code:: python

     >>> import torch
     >>> from vineyard.contrib.ml import pytorch
     >>> data_id = vineyard_client.put(dataset, typename='Tensor')
     >>> vin_data = vineyard_client.get(data_id)

The dataset object should be an object of the type CustomDataset class which is inherited 
from ``torch.utils.data.Dataset`` class. Adding the typename as ``Tensor`` is important. 
The ``vin_data`` will be of type ``torch.utils.data.TensorDataset``. 

Using Dataframe
^^^^^^^^^^^^^^^

.. code:: python

     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
     >>> label = torch.from_numpy(df['c'].values.astype(np.float32))
     >>> data = torch.from_numpy(df.drop('c', axis=1).values.astype(np.float32))
     >>> dataset = torch.utils.data.TensorDataset(data, label)
     >>> data_id = vineyard_client.put(dataset, typename='Dataframe', cols=['a', 'b', 'c'], label='c')
     >>> vin_data = vineyard_client.get(data_id, label='c)

While using the PyTorch form of the dataframe with vineyard, it is important to mention 
the typename as ``Dataframe``, a list of column names in ``cols`` and the ``label`` 
name in label tag. The ``vin_data`` will be of the form ``TensorDataset`` with 
the label as mentioned with the label tag. If no value is passed to the label tag 
vineyard will consider the default value which is the value of label passed in while 
calling the ``put`` method

Using RecordBatch of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> import pyarrow as pa
     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'f2'])
     >>> data_id = vineyard_client.put(batch)
     >>> vin_data = vineyard_client.get(data_id, label='f2')

The ``vin_data`` will be of the form ``TensorDataset`` with the label as mentioned 
with the label tag. In this case it is important to mention the label tag.

Using Tables of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'f2'])
     >>> batches = [batch]*3
     >>> table = pa.Table.from_batches(batches)
     >>> data_id = vineyard_client.put(table)
     >>> vin_data = vineyard_client.get(data_id, label='f2')

The ``vin_data`` object will be of the form ``TensorDataset`` with the label as mentioned 
with the label tag. In this case, it is important to mention the label tag.

MxNet
-----

Using Numpy Data
^^^^^^^^^^^^^^^^

Vineyard supports ``Array Datasets`` from the gluon.data of MxNet.

.. code:: python

     >>> import mxnet as mx
     >>> from vineyard.contrib.ml import mxnet
     >>> dataset = mx.gluon.data.ArrayDataset((data, label))
     >>> data_id = vineyard_client.put(dataset, typename='Tensor')
     >>> vin_data = vineyard_client.get(data_id)

The dataset object should be an object of the type ArrayDataset from ``mxnet.gluon.data`` 
class. Here, Adding the typename as ``Tensor`` is important. The ``vin_data`` will be 
of type ``mxnet.gluon.data.ArrayDataset``. 

Using Dataframe
^^^^^^^^^^^^^^^

.. code:: python

     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
     >>> label = df['c'].values.astype(np.float32)
     >>> data = df.drop('c', axis=1).values.astype(np.float32)
     >>> dataset = mx.gluon.data.ArrayDataset((data, label))
     >>> data_id = vineyard_client.put(dataset, typename='Dataframe', cols=['a', 'b', 'c'], label='c')
     >>> vin_data = vineyard_client.get(data_id, label='c)

While using the MxNet form of the dataframe with vineyard, it is important to mention 
the typename as ``Dataframe``, a list of column names in ``cols`` and the ``label`` 
name in label tag. The ``vin_data`` will be of the form ``ArrayDataset`` with 
the label as mentioned with the label tag. If no value is passed to the label tag 
vineyard will consider the default value which is the value of label passed in while 
calling the ``put`` method

Using RecordBatch of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> import pyarrow as pa
     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'f2'])
     >>> data_id = vineyard_client.put(batch)
     >>> vin_data = vineyard_client.get(data_id, label='f2')

The ``vin_data`` will be of the form ``ArrayDataset`` with the label as mentioned 
with the label tag. In this case, it is important to mention the label tag.

Using Tables of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'f2'])
     >>> batches = [batch]*3
     >>> table = pa.Table.from_batches(batches)
     >>> data_id = vineyard_client.put(table)
     >>> vin_data = vineyard_client.get(data_id, label='f2')

The ``vin_data`` object will be of the form ``ArrayDataset`` with the label as mentioned 
with the label tag. In this case, it is important to mention the label tag.

XGBoost
-------

Vineyard supports resolving ``XGBoost::DMatrix`` from various kinds of vineyard data types.

From Vineyard::Tensor
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arr = np.random.rand(4, 5)
     >>> vin_tensor_id = vineyard_client.put(arr)
     >>> dmatrix = vineyard_client.get(vin_tensor_id)

The ``dmatrix`` will be a ``DMatrix`` instance with the same shape ``(4, 5)`` resolved from the ``Vineyard::Tensor``
object with the id ``vin_tensor_id``.

From Vineyard::DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
     >>> vin_df_id = vineyard_client.put(df)
     >>> dmatrix = vineyard_client.get(vin_df_id, label='a')

The ``dmatrix`` will be a ``DMatrix`` instance with shape of ``(4, 2)`` and ``feature_names`` of ``['b', 'c']``.
While the label of ``dmatrix`` is the values of column ``a``.

Sometimes the dataframe is a complex data structure and only ``one`` column will be used as the ``features``.
We support this case by providing the ``data`` kwarg.

.. code:: python

     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 
     >>>                    'b': [[5, 1.0, 4], [6, 2.0, 3], [7, 3.0, 2], [8, 9.0, 1]]})
     >>> vin_df_id = vineyard_client.put(df)
     >>> dmatrix = vineyard_client.get(vin_df_id, data='b', label='a')

The ``dmatrix`` will have the shape of ``(4, 3)`` corresponding to the values of column ``b``.
While the label is the values of column ``a``.

From Vineyard::RecordBatch
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> import pyarrow as pa
     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'target'])
     >>> vin_rb_id = vineyard_client.put(batch)
     >>> dmatrix = vineyard_client.get(vin_rb_id, label='target')

The ``dmatrix`` will have the shape of ``(4, 2)`` and ``feature_names`` of ``['f0', 'f1']``.
While the label is the values of column ``target``.

From Vineyard::Table
^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arrays = [pa.array([1, 2]), pa.array([0, 1]), pa.array([0.1, 0.2])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'label', 'f2'])
     >>> batches = [batch] * 3
     >>> table = pa.Table.from_batches(batches)
     >>> vin_tab_id = vineyard_client.put(table)
     >>> dmatrix = vineyard_client.get(vin_tab_id, label='label')

The ``dmatrix`` will have the shape of ``(6, 2)`` and ``feature_names`` of ``['f0', 'f2']``.
While the label is the values of column ``label``.

Nvidia-DALI
-----------

Vineyard supports integration of ``Dali Pipelines``.

.. code:: python

     >>> from nvidia.dali import pipeline_def
     >>> pipeline = pipe(device_id=device_id, num_threads=num_threads, batch_size=batch_size)
     >>> pipeline.build()
     >>> pipe_out = pipeline.run()
     >>> data_id = vineyard_client.put(pipe_out)
     >>> vin_pipe = vineyard_client.get(data_id)

In this case, the pipe is a ``pipeline_def`` function. The data received after executing pipe.run() can
be stored into vineyard. The Pipeline should only return two values, namely data and label. The return 
type of the data and label values should be of type ``TensorList``. The ``vin_pipe`` object will be the 
output of a simple in-built pipeline after executing the pipeline.build() and pipeline.run(). It will 
simply return two values of type Pipeline.  