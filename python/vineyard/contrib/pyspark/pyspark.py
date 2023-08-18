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

import numpy as np
import pandas as pd

import pyspark
from pyspark.mllib.common import _java2py
from pyspark.sql import SparkSession

import vineyard
from vineyard.core import context
from vineyard.data.dataframe import make_global_dataframe


def pyspark_dataframe_builder(
    client, value, builder, **kw
):  # pylint: disable=unused-argument
    def put_partition(rdd):
        import os

        sock = os.environ.get('VINEYARD_IPC_SOCKET')
        rows = (row.asDict() for row in rdd)
        pandas_df = pd.DataFrame(rows)
        client = vineyard.connect(sock)
        obj_id = client.put(pandas_df)
        client.persist(obj_id)
        yield pd.DataFrame([{'id': int(obj_id)}])

    res = value.rdd.mapPartitions(put_partition).collect()
    blocks = [res[i]['id'][0] for i in range(len(res))]
    return make_global_dataframe(client, blocks)


def pyspark_dataframe_resolver(obj, resolver, **kw):  # pylint: disable=unused-argument
    def py4j_wrapper(obj_id, spark_conf, socket):
        sparkSession = SparkSession.builder.config(conf=spark_conf).getOrCreate()
        sc = sparkSession.sparkContext
        jvm = sc._jvm
        _jclient = jvm.io.v6d.core.client.IPCClient(socket)
        _jobjectId = jvm.io.v6d.core.common.util.ObjectID.fromString(obj_id.__repr__())
        _jmeta = _jclient.getMetaData(_jobjectId)
        _jvineyardRDD = jvm.io.v6d.spark.rdd.VineyardRDD(
            sc._jsc.sc(), _jmeta, "partitions_", socket, _jclient.getClusterStatus()
        )
        _jGlobalDataFrameRDD = jvm.io.v6d.spark.rdd.GlobalDataFrameRDD.fromVineyard(
            _jvineyardRDD
        )
        _jdf = _jGlobalDataFrameRDD.toDF(sparkSession._jsparkSession)
        df = _java2py(sc, _jdf)
        return df

    return py4j_wrapper(obj.meta.id, kw["spark_conf"], kw["socket"])


def register_pyspark_types(builder_ctx=None, resolver_ctx=None):
    if builder_ctx is not None:
        builder_ctx.register(pyspark.sql.dataframe.DataFrame, pyspark_dataframe_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::GlobalDataFrame', pyspark_dataframe_resolver)


@contextlib.contextmanager
def pyspark_context():
    with context() as (builder_ctx, resolver_ctx):
        register_pyspark_types(builder_ctx, resolver_ctx)
        yield builder_ctx, resolver_ctx
