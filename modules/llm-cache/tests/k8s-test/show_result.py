import re
import os
from kubernetes import client, config
import argparse

def get_pod_logs(api_instance, pod_name, namespace='default'):
    log = api_instance.read_namespaced_pod_log(name=pod_name, namespace=namespace)
    return log

def extract_last_metrics_from_log(log):
    #only reserve the last 5 lines
    log = log.split('\n')[-5:]
    log = '\n'.join(log)

    total_matched_tokens_re = re.compile(r"total matched tokens:  (\d+)")
    total_updated_tokens_re = re.compile(r"total updated tokens:  (\d+)")
    total_query_time_re = re.compile(r"total query time:  ([\d.]+)")
    total_update_time_re = re.compile(r"total update time:  ([\d.]+)")

    total_matched_tokens = total_matched_tokens_re.findall(log)
    total_query_time = total_query_time_re.findall(log)
    total_updated_tokens = total_updated_tokens_re.findall(log)
    total_update_time = total_update_time_re.findall(log)

    if len(total_matched_tokens) == 0 or len(total_query_time) == 0 or len(total_updated_tokens) == 0 or len(total_update_time) == 0:
        return None
    print(total_matched_tokens, total_query_time, total_updated_tokens, total_update_time)
    return int(total_matched_tokens[0]), float(total_query_time[0]), int(total_updated_tokens[0]), float(total_update_time[0])

def calculate_averages(metrics_list):
    total_count = len(metrics_list)
    if total_count == 0:
        return 0, 0

    total_query_speed = 0
    total_query_instance = 0
    total_update_speed = 0
    total_update_instance = 0
    max_update_speed = 0
    max_query_speed = 0

    for matched, query_time, updated, update_time in metrics_list:
        if matched != 0:
            if matched/query_time > max_query_speed:
                max_query_speed = matched/query_time
            total_query_speed += matched/query_time
            total_query_instance += 1
        if updated != 0:
            if updated/update_time > max_update_speed:
                max_update_speed = update_time / updated
            total_update_speed += updated/update_time
            total_update_instance += 1

    avg_matched_tokens_per_query_time = total_query_speed / total_query_instance
    avg_updated_tokens_per_update_time = total_update_speed / total_update_instance

    return avg_matched_tokens_per_query_time, avg_updated_tokens_per_update_time, max_query_speed, max_update_speed

def show_result(label_selector, kubeconfig_path):
    config.load_kube_config(config_file=os.path.expanduser(kubeconfig_path))
    api = client.CoreV1Api()

    pods = api.list_pod_for_all_namespaces(label_selector=label_selector)
    pod_metrics = []

    for pod in pods.items:
        log = get_pod_logs(api, pod.metadata.name, pod.metadata.namespace)
        metrics = extract_last_metrics_from_log(log)
        if metrics is not None:
            pod_metrics.append(metrics)

    average_metrics = calculate_averages(pod_metrics)
    print("Average matched tokens per query time:", average_metrics[0], " tokens/s")
    print("Average updated tokens per update time:", average_metrics[1], " tokens/s")
    print("Max query speed:", average_metrics[2], " tokens/s")
    print("Max update speed:", average_metrics[3], " tokens/s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-selector', type=str, default="app=fs-llm-test-worker")
    parser.add_argument('--kubeconfig-path', type=str, default='/your/kubeconfig')
    args = parser.parse_args()
    while True:
        show_result(args.label_selector,args.kubeconfig_path)
