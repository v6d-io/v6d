#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

''' This module exposes support for ObjectStream, that use can used like:

.. code:: python

    # create a builder, then seal it as stream
    >>> builder = ObjectStreamBuilder(client)
    >>> stream = builder.seal(client)
    >>> stream
    >>> <vineyard._C.ObjectStream at 0x13b3d7ef0>

    # use write to put ObjectIDs
    >>> writer = stream.open_writer(client)
    >>> df_id = client.put(pd.DataFrame(...))
    >>> writer.put_next(df_id)

    # mark the stream as finished
    >>> writer.finish()

    # open a reader
    >>> reader = stream.open_reader(client)
    >>> df_id = reader.get_next()
    >>> df_id
    ObjectID <"o00a5b58da21565a2">

    # the reader reaches the end of the stream
    >>> chunk = reader.get_next()
    ---------------------------------------------------------------------------
    StreamDrainedException                    Traceback (most recent call last)
    <ipython-input-20-d8809de11870> in <module>
    ----> 1 chunk = reader.next()

    StreamDrainedException: Stream drained: no more chunks
'''

from vineyard._C import ObjectStream, ObjectStreamBuilder, \
    ObjectStreamReader, ObjectStreamWriter
