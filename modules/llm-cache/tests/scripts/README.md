#### `dfs-fio.sh`

This script runs FIO on a kubernetes cluster to test the average bandwidth of a distributed file system. Assuming that you have deployed the persistent volume and persistent volume claim for the distributed file system, you can create a deployment with multiple replicas and mount the distributed file system to each replica. For more information on how to do this, see the [Mount Nas Volume on ACK](https://www.alibabacloud.com/help/en/ack/serverless-kubernetes/user-guide/mount-a-statically-provisioned-nas-volume).

Now, you may have many pods running on the kubernetes cluster, each of which has a mounted distributed file system. In general, each pod can be considered as a client of the distributed file system. With this script, you can test the average bandwidth of the distributed file system by running FIO on each pod and calculating the overall bandwidth.

> **Tip:** The average bandwidth may be bounded by the network bandwidth of the kubernetes nodes. If your kubernetes node is poor in network performance, the average bandwidth may be much lower than the theoretical bandwidth of the distributed file system (e.g., DFS overall bandwidth / pod replicas ).

Next, we will show how to run the script.

```bash
$ bash dfs-fio.sh   
Usage: dfs-fio.sh -l <pod-label> -a <fio-action> -j <num-jobs> -m <max-pods> -d <directory> -b <block-size>
  -l <pod-label>     Pod label for filtering specific Pods
  -a <fio-action>    FIO operation type: read, write, randread, randwrite
  -j <num-jobs>      Number of parallel jobs in FIO
  -m <max-pods>      Limit on the maximum number of Pods, should not exceed the total number of Pods
  -d <directory>     Directory path for FIO testing
  -b <block-size>    Block size for the FIO test
```

- Test the average **read** bandwidth for a pod.

  ```bash
  $ bash dfs-fio.sh -l your_pod_label_key=your_pod_label_value -a read -j 1 -m 1 -d /your/dfs/mountpath -b 4992k -n default
  ```

- Test the average **write** bandwidth for ten pods.

  ```bash
  $ bash dfs-fio.sh -l your_pod_label_key=your_pod_label_value -a write -j 1 -m 10 -d /your/dfs/mountpath -b 4992k -n default
  ```

- Test the average **randread** bandwidth

  ```bash
  $ bash dfs-fio.sh -l your_pod_label_key=your_pod_label_value -a randread -j 1 -m 10 -d /your/dfs/mountpath -b 4992k -n default
  ```

