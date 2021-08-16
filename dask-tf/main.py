import contextlib
import importlib
import json
import os
import platform
import socket
import subprocess
import time
import vineyard


import dask.array as da
import dask.dataframe as dd

import numpy as np
import pandas as pd
import tensorflow as tf

from dask import delayed
from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.dask.dask import register_dask_types

VINEYARD_CI_IPC_SOCKET = '/tmp/vineyard.ci.%s.sock' % time.time()


find_executable_generic = None
start_program_generic = None
find_port = None


def prepare_runner_environment():
    utils = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python', 'vineyard', 'deploy', 'utils.py')
    spec = importlib.util.spec_from_file_location("vineyard._contrib", utils)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    global find_executable_generic
    global start_program_generic
    global find_port
    find_executable_generic = getattr(mod, 'find_executable')
    start_program_generic = getattr(mod, 'start_program')
    find_port = getattr(mod, 'find_port')


prepare_runner_environment()


def find_executable(name):
    default_builder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'bin')
    binary_dir = os.environ.get('VINEYARD_EXECUTABLE_DIR', default_builder_dir)
    return find_executable_generic(name, search_paths=[binary_dir])


def start_program(*args, **kwargs):
    default_builder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'bin')
    binary_dir = os.environ.get('VINEYARD_EXECUTABLE_DIR', default_builder_dir)
    print('binary_dir = ', binary_dir)
    return start_program_generic(*args, search_paths=[binary_dir], **kwargs)


@contextlib.contextmanager
def start_etcd():
    with contextlib.ExitStack() as stack:
        client_port = find_port()
        peer_port = find_port()
        if platform.system() == 'Linux':
            data_dir_base = '/dev/shm'
        else:
            data_dir_base = '/tmp'
        proc = start_program('etcd',
                             '--data-dir', '%s/etcd-%s' % (data_dir_base, time.time()),
                             '--listen-peer-urls', 'http://0.0.0.0:%d' % peer_port,
                             '--listen-client-urls', 'http://0.0.0.0:%d' % client_port,
                             '--advertise-client-urls', 'http://127.0.0.1:%d' % client_port,
                             '--initial-cluster', 'default=http://127.0.0.1:%d' % peer_port,
                             '--initial-advertise-peer-urls', 'http://127.0.0.1:%d' % peer_port)
        yield stack.enter_context(proc), 'http://127.0.0.1:%d' % client_port


@contextlib.contextmanager
def start_vineyardd(etcd_endpoints, etcd_prefix, size=1 * 1024 * 1024 * 1024,
                    default_ipc_socket=VINEYARD_CI_IPC_SOCKET,
                    idx=None, **kw):
    rpc_socket_port = find_port()
    if idx is not None:
        socket = '%s.%d' % (default_ipc_socket, idx)
    else:
        socket = default_ipc_socket
    with contextlib.ExitStack() as stack:
        proc = start_program('vineyardd',
                             '--size', str(size),
                             '--socket', socket,
                             '--rpc_socket_port', str(rpc_socket_port),
                             '--etcd_endpoint', etcd_endpoints,
                             '--etcd_prefix', etcd_prefix,
                             verbose=True, **kw)
        yield stack.enter_context(proc), rpc_socket_port


@contextlib.contextmanager
def start_multiple_vineyardd(etcd_endpoints, etcd_prefix, size=2 * 1024 * 1024 * 1024,
                             default_ipc_socket=VINEYARD_CI_IPC_SOCKET,
                             instance_size=1, **kw):
    with contextlib.ExitStack() as stack:
        jobs = []
        for idx in range(instance_size):
            job = start_vineyardd(etcd_endpoints, etcd_prefix, size=size,
                                  default_ipc_socket=default_ipc_socket, idx=idx, **kw)
            jobs.append(job)
        yield [stack.enter_context(job) for job in jobs]

@contextlib.contextmanager
def launch_dask_cluster(vineyard_ipc_sockets, host, port):
    with contextlib.ExitStack() as stack:
        proc = start_program('dask-scheduler', '--host', host, '--port', str(port))
        stack.enter_context(proc)
        scheduler = f'tcp://{host}:{port}'
        clients = []
        workers = {}
        for sock in vineyard_ipc_sockets:
            client = vineyard.connect(sock)
            worker_name = 'dask_worker_%d' % client.instance_id
            workers[client.instance_id] = worker_name
            # launch a worker with corresponding name for each vineyard instance
            proc = start_program('dask-worker', scheduler, '--name', worker_name, verbose=True, VINEYARD_IPC_SOCKET=sock)
            stack.enter_context(proc)
            clients.append(client)
        yield clients, scheduler, workers


def dask_preprocess(vineyard_ipc_sockets):
    with launch_dask_cluster(vineyard_ipc_sockets, 'localhost', 8786) as dask_cluster:
        clients, dask_scheduler, _ = dask_cluster

        def get_mnist():
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            # The `x` arrays are in uint8 and have values in the [0, 255] range.
            # You need to convert them to float64 with values in the [0, 1] range.
            x_train = x_train / np.float64(255)
            y_train = y_train.astype(np.int64)
            return x_train.reshape(60000, 784), y_train

        datasets = [delayed(get_mnist)() for i in range(len(vineyard_ipc_sockets) * 2)]
        images = [d[0] for d in datasets]
        labels = [d[1] for d in datasets]

        images = [da.from_delayed(im, shape=(60000, 784), dtype='float64') for im in images]
        labels = [da.from_delayed(la, shape=(60000, 1), dtype='float64') for la in labels]

        images = da.concatenate(images, axis=0)
        labels = da.concatenate(labels, axis=0)
        
        images_id = clients[0].put(images, dask_scheduler=dask_scheduler)
        meta = clients[0].get_meta(images_id)
        assert meta['partitions_-size'] == len(vineyard_ipc_sockets) * 2

        labels_id = clients[0].put(labels, dask_scheduler=dask_scheduler)
        meta = clients[0].get_meta(labels_id)
        assert meta['partitions_-size'] == len(vineyard_ipc_sockets) * 2

        print('!!!!!!!!!!!!!!Success!!!!!!!!!!!!')

        return images_id, labels_id

@contextlib.contextmanager
def launch_tf_distributed(vineyard_ipc_sockets, x_id, y_id):
    with contextlib.ExitStack() as stack:
        workers = [f'localhost:1234{i}' for i in range(len(vineyard_ipc_sockets))]
        cfg = {'cluster': {'worker': workers}, 'task': {'type': 'worker', 'index': 0}}
        for idx, sock in enumerate(vineyard_ipc_sockets):
            cfg['task']['index'] = idx
            proc = start_program('python3', 'train.py', repr(x_id), repr(y_id), verbose=True, VINEYARD_IPC_SOCKET=sock, TF_CONFIG=json.dumps(cfg))
            stack.enter_context(proc)
        yield 'started'

def keras_train(vineyard_ipc_sockets, x_id, y_id):
    with launch_tf_distributed(vineyard_ipc_sockets, x_id, y_id) as _:
        time.sleep(1)
        print('.', end='')
        


def main():
    with start_etcd() as (_, etcd_endpoints):
        ipc_socket_tpl = '/tmp/vineyard.ci.dist.%s' % time.time()
        instance_size = 2
        etcd_prefix = 'vineyard_test_%s' % time.time()
        with start_multiple_vineyardd(etcd_endpoints,
                                    etcd_prefix,
                                    default_ipc_socket=ipc_socket_tpl,
                                    instance_size=instance_size,
                                    nowait=True) as instances:
            vineyard_ipc_sockets = ['%s.%d' % (ipc_socket_tpl, i) for i in range(instance_size)]
            with builder_context() as builder:
                with resolver_context() as resolver:
                    register_dask_types(builder, resolver)
                    x_id, y_id = dask_preprocess(vineyard_ipc_sockets)
                    keras_train(vineyard_ipc_sockets, x_id, y_id)
            

if __name__ == '__main__':
    main()
