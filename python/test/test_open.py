import sys
import filecmp

import vineyard
from vineyard.io.stream import open


def test_local_with_header(dir, socket):
    stream = open('file://%s/p2p-32.e#header_row=true&delimiter= ' % dir,
                  vineyard_ipc_socket=socket,
                  vineyard_endpoint='localhost:9607')
    open('file://%s/p2p-32.out' % dir, stream, mode='w', vineyard_ipc_socket=socket, vineyard_endpoint='localhost:9607')
    return filecmp.cmp('%s/p2p-32.e' % dir, '%s/p2p-32.out' % dir)


def test_local_without_header(dir, socket):
    stream = open('file://%s/p2p-31.e#header_row=false&delimiter= ' % dir,
                  vineyard_ipc_socket=socket,
                  vineyard_endpoint='localhost:9607')
    open('file://%s/p2p-31.out' % dir, stream, mode='w', vineyard_ipc_socket=socket, vineyard_endpoint='localhost:9607')
    return filecmp.cmp('%s/p2p-31.e' % dir, '%s/p2p-31.out' % dir)


def test_local_orc(dir, socket):
    stream = open('file://%s/test.orc' % dir, vineyard_ipc_socket=socket, vineyard_endpoint='localhost:9607')
    open('file://%s/testout.orc' % dir,
         stream,
         mode='w',
         vineyard_ipc_socket=socket,
         vineyard_endpoint='localhost:9607')
    return filecmp.cmp('%s/test.orc' % dir, '%s/testout.orc' % dir)


def test_hdfs_orc(dir, socket):
    stream = open('file://%s/test.orc' % dir, vineyard_ipc_socket=socket, vineyard_endpoint='localhost:9607')
    open('hdfs://dev:9000/tmp/testout.orc',
         stream,
         mode='w',
         vineyard_ipc_socket=socket,
         vineyard_endpoint='localhost:9607')
    streamout = open('hdfs://dev:9000/tmp/testout.orc', vineyard_ipc_socket=socket, vineyard_endpoint='localhost:9607')
    open('file://%s/testout1.orc' % dir,
         streamout,
         mode='w',
         vineyard_ipc_socket=socket,
         vineyard_endpoint='localhost:9607')
    return filecmp.cmp('%s/test.orc' % dir, '%s/testout1.orc' % dir)


def test_hive(dir, socket):
    stream = open('hive://dev:9000/user/hive/warehouse/pt',
                  vineyard_ipc_socket=socket,
                  vineyard_endpoint='localhost:9607')
    open('file://%s/testout1.e' % dir, stream, mode='w', vineyard_ipc_socket=socket, vineyard_endpoint='localhost:9607')
    return True


def test_hdfs_bytes(dir, socket):
    stream = open('file://%s/p2p-32.e#header_row=true&delimiter= ' % dir,
                  vineyard_ipc_socket=socket,
                  vineyard_endpoint='localhost:9607')
    open('hdfs://dev:9000/tmp/p2p-32.out',
         stream,
         mode='w',
         vineyard_ipc_socket=socket,
         vineyard_endpoint='localhost:9607')
    hdfs_stream = open('hdfs://dev:9000/tmp/p2p-32.out#header_row=true&delimiter= ',
                       vineyard_ipc_socket=socket,
                       vineyard_endpoint='localhost:9607')
    open('file://%s/p2p-32.out' % dir,
         hdfs_stream,
         mode='w',
         vineyard_ipc_socket=socket,
         vineyard_endpoint='localhost:9607')
    return filecmp.cmp('%s/p2p-32.e' % dir, '%s/p2p-32.out' % dir)


if __name__ == '__main__':
    local_test_dir = sys.argv[1]
    socket = sys.argv[2]
    result = (
        test_hdfs_bytes(local_test_dir, socket),
        test_hive(local_test_dir, socket),
        test_hdfs_orc(local_test_dir, socket),
        test_local_orc(local_test_dir, socket),
        test_local_without_header(local_test_dir, socket),
        test_local_with_header(local_test_dir, socket),
    )
    print(result)
