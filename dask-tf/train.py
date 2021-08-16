import os
import sys
import json

import tensorflow as tf
import mnist

per_worker_batch_size = 64
#index = os.environ.get('TF_WORKER_INDEX', None)
#tf_config = {'cluster': {'worker': ['localhost:12345', 'localhost:23456']}, 'task': {'type': 'worker', 'index': 0}}
#if index:
#  tf_config['task']['index'] = int(index)
#os.environ['TF_CONFIG'] = json.dumps(tf_config)

tf_config = json.loads(os.environ['TF_CONFIG'])
index = int(tf_config['task']['index'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

train_dataset = mnist.mnist_dataset(sys.argv[1], sys.argv[2], index, per_worker_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist.build_and_compile_cnn_model()


multi_worker_model.fit(train_dataset, epochs=3, steps_per_epoch=70)
