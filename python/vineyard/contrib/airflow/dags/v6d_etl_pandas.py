#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#
# Adapted from example dags in Airflow's documentation, see also
#
#  https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
#

import numpy as np
import pandas as pd

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
}


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2), tags=['example'])
def taskflow_etl_pandas():
    @task()
    def extract():
        order_data_dict = pd.DataFrame({'a': np.random.rand(100000), 'b': np.random.rand(100000)})
        return order_data_dict

    @task(multiple_outputs=True)
    def transform(order_data_dict: dict):
        return {"total_order_value": order_data_dict["a"].sum()}

    @task()
    def load(total_order_value: float):
        print(f"Total order value is: {total_order_value:.2f}")

    order_data = extract()
    order_summary = transform(order_data)
    load(order_summary["total_order_value"])


taskflow_etl_pandas_dag = taskflow_etl_pandas()
