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

# The data returned by this function then can be sent to a builder, to convert it to a vineyard object.