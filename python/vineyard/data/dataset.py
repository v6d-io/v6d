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

from vineyard._C import ObjectMeta
from torch.utils.data import Dataset
from .utils import from_json, to_json

def dataset_builder(client, value, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataSet'
    meta['num'] = to_json(len(value))
    for i in range(len(value)):
        data, label = value[i]
        meta.add_member(f'data_{i}_', builder.run(client, data))
        meta.add_member(f'label_{i}_', builder.run(client, label))
    return client.create_metadata(meta)

class VineyardDataSet(Dataset):
    '''
        Vineyard DataSet forked from PyTorch DataSet.
        Each item is a pair of data and label.
    '''
    def __init__(self, obj, resolver):
        '''
            obj is the object get from vineyard.
        '''
        self.obj = obj
        self.resolver = resolver
        self.num = from_json(obj.meta['num'])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        data = self.resolver.run(self.obj.member(f'data_{idx}_'))
        label = self.resolver.run(self.obj.member(f'label_{idx}_'))
        return (data, label)

def dataset_resolver(obj, resolver):
    return VineyardDataSet(obj, resolver)

def register_dataset_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(Dataset, dataset_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::DataSet', dataset_resolver)
