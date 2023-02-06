#! /usr/bin/env python3
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

import os
import time

import vineyard
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch import nn

from torch.utils.data import TensorDataset, DataLoader

def make_dataset_from_vineyard():
    ''' Prepare pytorch dataset using local dataframe chunks
    '''
    node_name = os.environ['NODENAME']
    client = vineyard.connect('/var/run/vineyard.sock')
    local_member_ids = os.environ.get(node_name, '').split(',')
    print(local_member_ids)

    chunks = []
    for local_id in local_member_ids:
        if local_id:
            chunk = client.get(vineyard.ObjectID(local_id))
            chunks.append(chunk)

    if not chunks:
        chunks.append(pd.DataFrame(np.zeros((0, 25))))
    merged_features = pd.concat(chunks)

    labels = merged_features.iloc[:, 2].to_numpy().astype('float32')
    features = merged_features.iloc[:, 3:].to_numpy().astype('float32')

    return TensorDataset(torch.from_numpy(features),
                         torch.from_numpy(labels))

def fit(dataset, model, loss_fn, opt, num_epochs=5, batch_size=5):
    train_dl = DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for xdata, ylabel in train_dl:
            pred = model(xdata)
            loss = loss_fn(pred, ylabel)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('tranning for epoch: %s' % epoch, flush=True)


def training():
    start = time.time()
    model = nn.Linear(24, 1)
    model = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    loss_fn = nn.functional.mse_loss
    dataset = make_dataset_from_vineyard()
    if dataset:
        fit(dataset, model, loss_fn, opt)

    print('training model usage: %s' % (time.time() - start), flush=True)

socket = '/var/run/vineyard.sock'
vineyard_client = vineyard.connect(socket)
instance_id = vineyard_client.instance_id
env_dist = os.environ
allinstances = int(env_dist['ALLINSTANCES'])
rank = instance_id%allinstances
world_size = int(os.environ.get('WORLD_SIZE', 1))

print('rank is',rank,flush=True)
print('start training...', flush=True)
dist.init_process_group("gloo", world_size=world_size, rank=rank)
training()
dist.destroy_process_group()
print('test passed',flush=True)

time.sleep(600)
