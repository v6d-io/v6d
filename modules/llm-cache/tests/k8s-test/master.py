import socket
import random
import os
import time
from kubernetes import client, config
from multiprocessing import Pool

def get_pod_ips(label_selector):
    config.load_incluster_config()
    api = client.CoreV1Api()
    pods = api.list_pod_for_all_namespaces(label_selector=label_selector)
    pod_ip_list = []
    for pod in pods.items:
        pod_ip_list.append(pod.status.pod_ip)
    return pod_ip_list

def distribute_prompts(args):
    file_name, server_ips = args
    token_list = []
    with open(f'{file_name}', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            token_list.append(line)

    for token in token_list:
        server_ip = random.choice(server_ips)
        #time.sleep(random.randint(1, 200)/1000000)
        while True:
            try:
                send_tokens_to_server(server_ip, 8888, token)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
                continue

def send_tokens_to_server(server_address, server_port, tokens):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect((server_address, server_port))
    clientsocket.send(tokens.encode('utf-8'))
    clientsocket.close()

if __name__ == "__main__":
    file_num = int(os.environ.get('TOKENS_FILE_NUM', 16))
    file_names = [f'/tokens/tokens_{i}' for i in range(file_num)]
    pod_selector = os.environ.get('POD_SELECTOR', 'app=fs-llm-test-worker')
    server_ips = get_pod_ips(pod_selector)
    with Pool(file_num) as p:
        p.map(distribute_prompts, [(file_name, server_ips) for file_name in file_names])
