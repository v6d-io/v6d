from vineyard._C import ObjectMeta
from .utils import from_json, to_json

import tensorflow as tf

# This function will support a common dataset type with x and y parameters.
# Various methods can be included to the function to make is more flexible
# Tensorflow Documentation: https://www.tensorflow.org/api_docs/python/tf/data/Dataset

def tf_dataset(x, y, is_map=False, is_cache=False, is_shuffle=False,
                map_func=None, batch_vars=32, prefetch_buffer_size=64, shuffle_buffer_size=10000, **kwargs):
    data = tf.data.Dataset.from_tensor_slices((x,y))
    if is_shuffle:
        data = data.shuffle(shuffle_buffer_size)
    if is_map:
        data = data.map(map_func)
    data = data.batch(batch_vars)
    if is_cache:
        data = data.cache()
    data = data.prefetch(prefetch_buffer_size)
    return data

def dataset_builder(client, value, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::TfDataSet'
    meta['num'] = to_json(len(value))
    for i in range(len(value)):
        data, label = value[i]
        meta.add_member(f'data_{i}_', builder.run(client, data))
        meta.add_member(f'label_{i}_', builder.run(client, label))
    return client.create_metadata(meta)

def dataset_resolver(obj, resolver):
    meta = obj.meta
    num = from_json(meta['num'])
    x = []
    y = []
    for i in range(num):
        data = resolver.run(obj.member(f'data_{i}_'))
        label = resolver.run(obj.member(f'label_{i}_'))
        x.append(data)
        y.append(label)
    return tf.data.Dataset.from_tensor_slices((x,y))


def register_dataset_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(tf.data.Dataset, dataset_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::TfDataSet', dataset_resolver)
