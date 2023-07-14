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

import os

from kedro.runner import ParallelRunner as KedroParallelRunner

from vineyard.contrib.kedro.io import VineyardDataSet


class ParallelRunner(KedroParallelRunner):
    def create_default_data_set(self, ds_name: str) -> VineyardDataSet:
        '''Factory method for creating the default dataset for the runner.

        Args:
            ds_name: Name of the missing dataset.

        Returns:
            An instance of ``VineyardDataSet`` to be used for all
            unregistered datasets.
        '''
        return VineyardDataSet(ds_name=ds_name)
