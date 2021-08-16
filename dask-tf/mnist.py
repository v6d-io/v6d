import os
import vineyard
import tensorflow as tf
import numpy as np
from vineyard.core.resolver import resolver_context

def numpy_resolver(obj, resolver, **kw):
    print(obj.meta)
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_-{i}'].islocal:
            data.append(resolver.run(obj.member(f'partitions_-{i}')))
    print('-'*40, len(data))
    return np.concatenate(data)

def mnist_dataset(x_id, y_id, index, batch_size):
  with resolver_context({'vineyard::GlobalTensor': numpy_resolver}) as resolver:
    client = vineyard.connect()
    x_train = client.get(x_id)
    y_train = client.get(y_id)
    print('-'*40, x_train.shape, y_train.shape)
    train_datasets = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).repeat().batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_datasets_no_auto_shard = train_datasets.with_options(options)

    return train_datasets_no_auto_shard

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model