#!/bin/bash

# Usage instructions
usage() {
  echo "Usage: $0 -l <pod-label> -a <fio-action> -j <num-jobs> -m <max-pods> -d <directory> -b <block-size> -n <namespace>"
  echo "  -l <pod-label>     Pod label for filtering specific Pods"
  echo "  -a <fio-action>    FIO operation type: read, write, randread, randwrite"
  echo "  -j <num-jobs>      Number of parallel jobs in FIO"
  echo "  -m <max-pods>      Limit on the maximum number of Pods, should not exceed the total number of Pods"
  echo "  -d <directory>     Directory path for FIO testing"
  echo "  -b <block-size>    Block size for the FIO test"
  echo "  -n <namespace>     Namespace for the Pods"
  exit 1
}

# Parse command line options
while getopts "l:a:j:m:d:b:n:" opt; do
  case ${opt} in
    l ) pod_label=$OPTARG ;;
    a ) fio_action=$OPTARG ;;
    j ) num_jobs=$OPTARG ;;
    m ) max_pods=$OPTARG ;;
    d ) directory=$OPTARG ;;
    b ) block_size=$OPTARG ;;
    n ) namespace=$OPTARG ;;
    ? ) usage ;;
  esac
done

# Verify that all required parameters have been provided
if [ -z "${pod_label}" ] || [ -z "${fio_action}" ] || [ -z "${num_jobs}" ] || [ -z "${max_pods}" ] || [ -z "${directory}" ] || [ -z "${block_size}" ] || [ -z "${namespace}" ]; then
    usage
fi

# Get all Pod names with the specified label
pods=$(kubectl get po -lapp=${pod_label} -o custom-columns=:metadata.name)

# Create a directory to store log files
mkdir -p fio_logs

counter=0

# Run FIO in parallel and collect the results
for pod in $pods; do
  if [ $counter -ge $max_pods ]; then
    echo "Reached the maximum number of pods ($max_pods). Exiting loop."
    break
  fi
  echo "Executing fio command on pod: $pod"
  kubectl exec -n ${namespace} "$pod" -- fio -direct=1 -directory=${directory} -ioengine=libaio -iodepth=1024 -rw=${fio_action} -bs=${block_size} -size=5G -numjobs=${num_jobs} -time_based=1 -group_reporting -name="${pod}_${fio_action}_Testing" > "fio_logs/$pod.log" &
  ((counter++))
done

# Wait for all background jobs to complete
wait
echo "FIO testing has completed on all pods."

# Initialize total bandwidth and counter
total_bw_MiB=0
count=0

# Extract and accumulate bandwidth values
for file in fio_logs/*.log; do
  bw_MiB=$(grep -oP 'bw=\K[0-9]+(?=MiB/s)' "$file" | head -n1)
  if [[ ! -z "$bw_MiB" ]]; then
    total_bw_MiB=$(echo "$total_bw_MiB + $bw_MiB" | bc)
    count=$((count + 1))
  fi
done

# If there are log entries processed, calculate the average; otherwise, indicate no valid logs
if [ $count -gt 0 ]; then
  average_bw_MiB=$(echo "scale=2; $total_bw_MiB / $count" | bc)
  echo "Average Bandwidth: $average_bw_MiB MiB/s"
else
  echo "No valid log entries found."
fi

# Clean up the log folder
rm -rf fio_logs
