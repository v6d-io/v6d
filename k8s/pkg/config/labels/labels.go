/** Copyright 2020-2023 Alibaba Group Holding Limited.

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
	/* following labels are used for speficifying the vineyard components */

	// VineyardAppLabel is the special label key for vineyard application
	VineyardAppLabel = "app.vineyard.io/name"

	// VineyardRoleLabel is the label key for vineyardd role
	VineyardRoleLabel = "app.vineyard.io/role"
	/* following labels are used for scheduling */

	// SchedulingEnabledLabel is the label key for enabling scheduling
	SchedulingEnabledLabel = "scheduling.v6d.io/enabled"

	// VineyardObjectJobLabel is the label key to indicate the job of the vineyard object
	VineyardObjectJobLabel = "k8s.v6d.io/job"

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

	/* following labels are used for operation injection */

	// AssemblyEnabledLabel is the label for assembly, and inject the assembly container when setting true
	AssemblyEnabledLabel = "assembly.v6d.io/enabled"
	// RepartitionEnabledLabel is the label for repartition, and inject the repartition container when setting true
	RepartitionEnabledLabel = "repartition.v6d.io/enabled"

	/* following labels are used for sidecar injection */

	// SidecarEnableLabel is the label key for enabling sidecar injection
	SidecarEnableLabel = "sidecar-injection"
)
