import time
import socket
import os
import argparse

import numpy as np

from vineyard.llm import KVCache, KVTensor
from vineyard.llm.config import FileCacheConfig

def start_server(port=8888):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = os.environ.get('POD_IP', 'localhost')
    layer = int(os.environ.get('LAYER', 96))
    batch_size = int(os.environ.get('BATCH_SIZE', 16))
    split_number = int(os.environ.get('SPLIT_NUMBER', 2))
    cache_path = os.environ.get('CACHE_PATH', '/mnt/llm_cache')
    client_gc_interval = int(os.environ.get('CLIENT_GC_INTERVAL', 30 * 60))
    ttl = int(os.environ.get('TTL', 30 * 60))
    enable_global_gc = os.environ.get('ENABLE_GLOBAL_GC', False).lower() in ['true', '1']
    global_gc_interval = int(os.environ.get('GLOBAL_GC_INTERVAL', 3 * 60 * 60))
    global_ttl = int(os.environ.get('GLOBAL_TTL', 3 * 60 * 60))
    serversocket.bind((ip, port))
    serversocket.listen(20)

    kv_tensor_shape = (104, 128)
    kv_tensor_dtype = np.float16
    kv_tensor = np.random.randn(*kv_tensor_shape).astype(kv_tensor_dtype)

    file_cache_config = FileCacheConfig(
        chunk_size = int(batch_size),
        split_number = int(split_number),
        root = cache_path,
        client_gc_interval = client_gc_interval,
        ttl = ttl,
        enable_global_gc = enable_global_gc,
        global_gc_interval = global_gc_interval,
        global_ttl = global_ttl,
    )
    cache = KVCache(
        cache_config = file_cache_config,
        tensor_bytes=kv_tensor.nbytes,  # should be the same as the nbytes of the tensor
        cache_capacity=1024,
        layer=int(layer),
    )

    total_matched_tokens = 0
    total_updated_tokens = 0
    total_tokens = 0
    total_update_time = 0
    total_query_time = 0


    def reserve_kv_tensors(kv_tensors, num_tokens, kv_tensor):
        num_to_reserve = num_tokens - len(kv_tensors)
        if num_to_reserve <= 0:
            return kv_tensors
        for _ in range(num_to_reserve):
            kv_tensors.append([
                (KVTensor(kv_tensor.ctypes.data, kv_tensor.nbytes),
                 KVTensor(kv_tensor.ctypes.data, kv_tensor.nbytes))
                for _ in range(layer)
            ])
        return kv_tensors

    # used to hold the query results
    kv_state_list = []

    while True:
        clientsocket, _ = serversocket.accept()

        tokens = b''
        while True:
            data = clientsocket.recv(1024)
            if not data:
                break
            tokens += data

        tokens = tokens.decode('utf-8')
        tokens = tokens.replace('\n', '').split(' ')
        tokens = [int(token) for token in tokens]

        kv_state_list = reserve_kv_tensors(kv_state_list, len(tokens), kv_tensor)

        query_start_time = time.time()
        matched = cache.query(tokens, kv_state_list)
        query_end_time = time.time()
        if matched > 0:
            total_query_time += query_end_time - query_start_time

        total_matched_tokens += matched
        total_tokens += len(tokens)

        remaining = tokens[matched:]
        kv_state_list_remaining = [
            [ (KVTensor(kv_tensor.ctypes.data, kv_tensor.nbytes),
              KVTensor(kv_tensor.ctypes.data, kv_tensor.nbytes))
              for _ in range(layer)
            ] for _ in remaining
        ]
        update_start_time = time.time()
        updated = cache.update(tokens[:matched], remaining, kv_state_list_remaining)
        total_updated_tokens += updated
        update_end_time = time.time()
        if updated > 0:
            total_update_time += update_end_time - update_start_time

        print("matched tokens: ", matched, " / ", len(tokens), flush=True)
        print("query time: ", query_end_time - query_start_time, flush=True)
        print("update time: ", update_end_time - update_start_time, flush=True)
        print("total matched tokens: ", total_matched_tokens, " / ", total_tokens, flush=True)
        print("total updated tokens: ", total_updated_tokens, " / ", total_tokens, flush=True)
        print("total query time: ", total_query_time, flush=True)
        print("total update time: ", total_update_time, flush=True)
        clientsocket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    start_server(args.port)
