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
     >>> data_id = client.put(dataset)
     >>> vin_data = client.get(data_id)
     
The ``vin_data`` will be a shared-memory object from the vineyard.  

Using Dataframe
^^^^^^^^^^^^^^^

.. code:: python

     >>> import pandas as pd
     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'target': [1.0, 2.0, 3.0, 4.0]})
     >>> label = df.pop('target') 
     >>> dataset = tf.data.Dataset.from_tensor_slices((dict(df), label))
     >>> data_id = client.put(dataset)
     >>> vin_data = client.get(data_id)

Wrap the dataframe with ``tf.data``. This enables the use of feature columns as a 
bridge to map from the columns in Pandas Dataframe to features in Dataset. The 
dataset should return a dictionary of column names (from the dataframe) that map 
to column values. The dataset should only contain ``numerical data``. 

Using RecordBatch of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: python

     >>> import pyarrow as pa
     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'label'])
     >>> data_id = client.put(batch)
     >>> vin_data = client.get(data_id)

Vineyard support direct integration of RecordBatch. The ``vin_data`` object will 
be a tensorflow dataset. Here the ``label`` row should be named as ``label``.

Using Tables of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'label'])
     >>> batches = [batch]*3
     >>> table = pa.Table.from_batches(batches)
     >>> data_id = client.put(table)
     >>> vin_data = client.get(data_id)

Vineyard support direct integration of Tables as well. Here, the ``vin_data`` 
object will be of type tensorflow dataset. Here the ``label`` row should 
be named as ``label``.


PyTorch
-------

Using Numpy Data
^^^^^^^^^^^^^^^^

Vineyard support ``Custom Datasets`` inherited from the PyTorch Dataset.

.. code:: python

     >>> import torch
     >>> from vineyard.contrib.ml import pytorch
     >>> data_id = client.put(dataset, typename='Tensor')
     >>> vin_data = client.get(data_id)

The dataset object should be an object of the CustomDataset class inherited from 
``torch.utils.data.Dataset``. Adding the typename as ``Tensor`` is important. The 
``vin_data`` will be of type ``torch.utils.data.TensorDataset``. 

Using Dataframe
^^^^^^^^^^^^^^^

.. code:: python

     >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
     >>> label = torch.tensor(df['c'].values.astype(np.float32))
     >>> data = torch.tensor(df.drop('c', axis=1).values.astype(np.float32))
     >>> dataset = torch.utils.data.TensorDataset(data, label)
     >>> data_id = client.put(dataset, typename='Dataframe', cols=['a', 'b', 'c'], label='c')
     >>> vin_data = client.get(data_id, label='c)

While using the PyTorch from of dataframe with vineyard, it is important to mention 
the typename as ``Dataframe``, a list of column names in ``cols`` and the ``label`` 
name in label tag. The ``vin_data`` will be of form ``TensorDataset`` with 
the label as mentioned with the label tag. If no value is passed to the label tag 
it will consider the default value which is the value of label passed in while 
calling the ``put`` method

Using RecordBatch of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> import pyarrow as pa
     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'f2'])
     >>> data_id = client.put(batch)
     >>> vin_data = client.get(data_id, label='f2')

The ``vin_data`` will be of the form ``TensorDataset`` with the label as mentioned 
with the label tag. Here it is important to mention the label tag.

Using Tables of Pyarrow
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

     >>> arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
     >>> batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'f2'])
     >>> batches = [batch]*3
     >>> table = pa.Table.from_batches(batches)
     >>> data_id = client.put(table)
     >>> vin_data = client.get(data_id, label='f2')

The ``vin_data`` object will be of the form ``TensorDataset`` with the label as mentioned 
with the label tag. Here it is important to mention the label tag.

XGBoost
-------

