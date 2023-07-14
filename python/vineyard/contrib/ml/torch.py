#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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

import contextlib
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa

import lazy_import

from vineyard._C import ObjectMeta
from vineyard.core import context
from vineyard.data.utils import to_json

torch = lazy_import.lazy_module("torch")
torchdata = lazy_import.lazy_module("torchdata")


class WholeBatchSampler(torch.utils.data.Sampler[List[int]]):
    r"""Wraps another sampler to yield a single batch of all indices.

    Args:
        sampler (Dataset, Sampler or Iterable): Dataset, or the base sampler (can be any
                                                iterable object).

    Example:
        >>> list(WholeBatchSampler(SequentialSampler(range(10))))
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    """

    def __init__(
        self,
        sampler: Union[
            torch.utils.data.Dataset, torch.utils.data.Sampler[int], Iterable[int]
        ],
    ) -> None:
        if isinstance(sampler, torch.utils.data.Dataset):
            sampler = torch.utils.data.SequentialSampler(sampler)
        self.sampler = sampler

    def __iter__(self) -> Iterator[List[int]]:
        return iter([list(self.sampler)])

    def __len__(self) -> int:
        return 1


def torch_tensor_builder(client, value, builder, **kw):
    return builder.run(client, value.numpy(), **kw)


def torch_dataset_builder(client, value, builder, **kw):
    dsl = torch.utils.data.DataLoader(value, batch_sampler=WholeBatchSampler(value))

    # the dataloader will contains only one batch
    columns = next(iter(dsl))

    # build as a dataframe
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    meta['columns_'] = to_json(list(range(len(columns))))
    meta.add_member('index_', builder.run(client, pd.RangeIndex(columns[0].shape[0])))

    for index, column in enumerate(columns):
        meta['__values_-key-%d' % index] = to_json(index)
        meta.add_member(
            '__values_-value-%d' % index, builder.run(client, column.numpy(), **kw)
        )
    meta['nbytes'] = 0  # FIXME
    meta['__values_-size'] = len(columns)
    meta['partition_index_row_'] = kw.get('partition_index', [-1, -1])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [-1, -1])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def torch_tensor_resolver(obj, resolver, **kw):
    value = resolver.parent_context.run(obj, **kw)
    return torch.tensor(value)


def torch_dataset_resolver(obj, resolver, **kw):
    value = resolver.parent_context.run(obj, **kw)
    if isinstance(value, pd.DataFrame):
        return torch.utils.data.TensorDataset(
            *[torch.tensor(np.array(value[column].values)) for column in value.columns]
        )
    elif isinstance(value, (pa.Table, pa.RecordBatch)):
        return torch.utils.data.TensorDataset(
            *[torch.tensor(column.to_numpy()) for column in value.columns]
        )
    else:
        raise TypeError(f'torch dataset: unsupported type {type(value)}')


def torch_global_tensor_resolver(obj, resolver, **_kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_{i}'].islocal:
            data.append(
                torch.utils.data.TensorDataset(
                    resolver.run(obj.member(f'partitions_{i}'))
                )
            )
    return torch.utils.data.ConcatDataset(data)


def torch_global_dataframe_resolver(obj, resolver, **_kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_{i}'].islocal:
            data.append(resolver.run(obj.member(f'partitions_{i}')))
    return torch.utils.data.ConcatDataset(data)


def register_torch_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(torch.Tensor, torch_tensor_builder)
        builder_ctx.register(torch.utils.data.Dataset, torch_dataset_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', torch_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', torch_dataset_resolver)
        resolver_ctx.register('vineyard::RecordBatch', torch_dataset_resolver)
        resolver_ctx.register('vineyard::Table', torch_dataset_resolver)
        resolver_ctx.register('vineyard::GlobalTensor', torch_global_tensor_resolver)
        resolver_ctx.register(
            'vineyard::GlobalDataFrame', torch_global_dataframe_resolver
        )


def datapipe(
    dataset: torch.utils.data.Dataset,
) -> torchdata.datapipes.iter.IterableWrapper:
    '''Convert a torch.utils.data.Dataset to a torchdata.datapipes.iter.IterableWrapper.

        e.g.,

        .. code:: python

            with torch_context():
                # using existing vineyard object as the dataset
                ds = client.get(object_id)

                # convert to datapipes
                ds = datapipe(ds)

                # do some transformation
                ds2 = ds.map(lambda x: x + 1)

                # iterator
                for index, record in enumerate(ds2):
                    print(record)

    Args:
        dataset: The torch.utils.data.Dataset to be converted.

    Returns:
        A torchdata.datapipes.iter.IterableWrapper.
    '''
    return torchdata.datapipes.iter.IterableWrapper(dataset)


@contextlib.contextmanager
def torch_context():
    with context() as (builder_ctx, resolver_ctx):
        with contextlib.suppress(ImportError):
            register_torch_types(builder_ctx, resolver_ctx)
        yield builder_ctx, resolver_ctx
