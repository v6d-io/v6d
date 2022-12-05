/** Copyright 2020-2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package labels contains all labels provided by vineyard-operator for users
package labels

const (
	// VineyardJobName is the pod group name
	VineyardJobName = "scheduling.k8s.v6d.io/job"
	// VineyardJobReplica is the replication of pods in this job.
	VineyardJobReplica = "scheduling.k8s.v6d.io/replica"

	// VineyarddNamespace is the namespace of vineyardd
	VineyarddNamespace = "scheduling.k8s.v6d.io/vineyardd-namespace"
	// VineyarddName is the name of the vineyardd
	VineyarddName = "scheduling.k8s.v6d.io/vineyardd"
	// WorkloadReplicas is the replicas of workload, for dask repartition here
	WorkloadReplicas = "scheduling.k8s.v6d.io/replicas"
)
