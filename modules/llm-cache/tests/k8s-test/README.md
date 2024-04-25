## Run llm test on k8s

This document describes how to run the llm test on a Kubernetes cluster.

### Tokenize the prompt file

Suppose you have a [prompt file](./prompt-samples.txt) that contains the conversation info between the user and the chatbot. You can tokenize the prompt file by running the following command:

```bash
$ python tokenize_prompt.py --prompt-file prompt-samples.txt --file-num 1
```

After running the command, you will get a tokenized prompt file named `tokens_0` under the `small_files` directory.

```bash
$ ls small_files 
prompts_0.txt  tokens_0
```

Also, you could set the `--file-num` to the number of files you want to split the prompt file into. If the prompt file is too large, you can split it into multiple files. Each file will be processed in parallel.

```
$ python tokenize_prompt.py --prompt-file prompt-samples.txt --file-num 2
$ ls small_files
prompts_0.txt  prompts_1.txt  tokens_0  tokens_1
```

At this point, you can put these token files to the OSS bucket or NAS refer to the [ossutil upload files](https://help.aliyun.com/zh/oss/user-guide/upload-objects-to-oss/?spm=a2c4g.11186623.0.0.4b471c22sHG1EG) or [nas mount](https://help.aliyun.com/zh/nas/user-guide/mount-an-nfs-file-system-on-a-linux-ecs-instance?spm=a2c4g.11186623.0.0.15713eedDgiEYF).

### Build the master and worker images

Before building the master and worker images, you need to build the vineyard-python-dev image first, as we need the llm-cache pypi package.

```bash
$ cd v6d && make -C docker vineyard-python-dev
```

Then, you can build the master and worker images by running the following command:

>  Make sure the image registry is set correctly.

```bash
$ cd modules/llm-cache/tests/k8s-test
$ make build-images
```

Next, push the images to the registry:

```bash
$ make push-images
```

### Deploy on the k8s cluster

#### Create the OSS volume

Assume we have put the token files to the OSS bucket, we need to [create the oss secret and oss volume](https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/user-guide/mount-statically-provisioned-oss-volumes#title-hos-c75-12q) first.

#### Create the Distributed FileSystem Volume

The DFS could be NAS or CPFS, you could refer to the [Mount Nas Volume on ACK](https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/user-guide/mount-statically-provisioned-nas-volumes?spm=a2c4g.11186623.0.0.b7c130b7eJHcnf) or [Mount CPFS Volume on ACK](https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/user-guide/statically-provisioned-cpfs-2-0-volumes-1?spm=a2c4g.11186623.0.0.399a22dbapWWsP) to create the volume.

#### Deploy the worker

After preparing the OSS volume, and DFS volume, you need change the NFS volume name `nas-csi-pvc` to the DFS volume you created before.

> ** Note: ** The CPU resources is important for the performance of worker, you could adjust the `resources.requests.cpu` to get better performance.

Then deploy the worker by running the following command:

```bash
$ kubectl apply -f yamls/worker.yaml
```

#### Deploy the master

After deploying the worker, you need to change `TOKENS_FILE_NUM` environment variable in the `yamls/master.yaml` file to the number of token files you put in the OSS bucket. Also, the OSS VolumeClaimName `oss-pvc` should be set to the OSS volume you created.

Then deploy the master by running the following command:

```bash
$ kubectl apply -f yamls/master.yaml
```

### Show the result

After running the llm test, you can check the result by running the following command:

```bash
$ python show_result.py --kubeconfig-path /your/kubeconfig --label-selector your_label_key=your_label_value
```
