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

import pandas as pd

import pyspark.sql.functions as F
import pytest
from pyspark import SparkConf
from pyspark.mllib.random import RandomRDDs
from pyspark.sql import SparkSession

import vineyard
from vineyard.contrib.pyspark.pyspark import pyspark_context
from vineyard.data.dataframe import make_global_dataframe
from vineyard.deploy.utils import start_program


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_pyspark():
    with pyspark_context():
        yield


@contextlib.contextmanager
def launch_pyspark_cluster(vineyard_ipc_socket, host, port, vineyard_spark_jar_path):
    with contextlib.ExitStack() as stack:
        proc = start_program(
            "start-master.sh",
            "--host",
            host,
            "--port",
            str(port),
            verbose=False,
        )

        stack.enter_context(proc)
        master = f'spark://{host}:{port}'
        nworkers = 4
        proc = start_program(
            "start-worker.sh", master, SPARK_WORKER_INSTANCES=nworkers, verbose=False
        )
        stack.enter_context(proc)
        conf = (
            SparkConf()
            .setAppName('pyspark-vineyard-test')
            .setMaster(master)
            .set('spark.executorEnv.VINEYARD_IPC_SOCKET', vineyard_ipc_socket)
            .set('spark.jars', vineyard_spark_jar_path)
        )
        yield conf, vineyard_ipc_socket


@pytest.fixture(scope="module", autouse=True)
def pyspark_cluster(vineyard_ipc_socket, vineyard_spark_jar_path):
    with launch_pyspark_cluster(
        vineyard_ipc_socket, 'localhost', 7077, vineyard_spark_jar_path
    ) as cluster:
        yield cluster


def test_pyspark_dataframe_builder(pyspark_cluster):
    conf, sock = pyspark_cluster
    client = vineyard.connect(sock)
    sparkSession = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = sparkSession.sparkContext
    df = (
        RandomRDDs.uniformVectorRDD(sc, 1024, 1024)
        .map(lambda a: a.tolist())
        .toDF()
        .repartition(4)
    )
    obj_id = client.put(df)
    meta = client.get_meta(obj_id)
    assert meta['partitions_-size'] == 4


def test_pyspark_dataframe_resolver(pyspark_cluster):
    conf, sock = pyspark_cluster
    client = vineyard.connect(sock)
    chunks = []
    for i in range(4):
        chunk = client.put(pd.DataFrame({'x': [i, i * 2], 'y': [i * 3, i * 4]}))
        client.persist(chunk)
        chunks.append(chunk)

    gdf = make_global_dataframe(client, chunks)
    pdf = client.get(gdf.id, spark_conf=conf, socket=sock)
    assert pdf.select(F.sum('x'), F.sum('y')).rdd.map(sum).collect()[0] == 60
