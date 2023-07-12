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

import logging

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.io import DataSetNotFoundError
from kedro.pipeline import Pipeline

from vineyard.contrib.kedro.io import VineyardDataSet

logger = logging.getLogger('vineyard')


class VineyardHook:
    @hook_impl
    def before_pipeline_run(self, pipeline: Pipeline, catalog: DataCatalog) -> None:
        self.catalog = catalog.shallow_copy()

        # add unregistered datasets, see also
        #
        #  - https://github.com/kedro-org/kedro/blob/main/kedro/runner/runner.py#L47
        for ds_name in pipeline.data_sets() - set(catalog.list()):
            catalog.add(ds_name, VineyardDataSet(ds_name=ds_name))

        # replace intermediate datasets with vineyard datasets
        self.intermediate_data_sets = (
            pipeline.data_sets() - pipeline.inputs() - pipeline.outputs()
        )
        replaced = dict()
        for ds_name in self.intermediate_data_sets:
            try:
                dataset = catalog._get_dataset(ds_name)
                if not isinstance(dataset, VineyardDataSet):
                    replaced[ds_name] = VineyardDataSet(ds_name=ds_name)
            except DataSetNotFoundError:
                continue
        catalog.add_all(replaced, replace=True)


hooks = VineyardHook()
