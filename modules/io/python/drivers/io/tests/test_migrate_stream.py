import filecmp
import itertools
import pytest
import configparser
import glob
import os
from urllib.parse import urlparse

import vineyard
import vineyard.io

@pytest.mark.skip_without_migration()
def test_migrate_stream(vineyard_ipc_sockets, vineyard_endpoint, test_dataset):
    vineyard_ipc_sockets = list(itertools.islice(itertools.cycle(vineyard_ipc_sockets), 2))

    stream = vineyard.io.open(
        "file://%s/p2p-31.e" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_sockets[0],
        vineyard_endpoint=vineyard_endpoint,
        read_options={
            "header_row": False,
            "delimiter": " "
        },
    )

    client = vineyard.connect(vineyard_ipc_sockets[1])
    new_stream = client.migrate_stream(stream.id)

    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        new_stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp("%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp)
