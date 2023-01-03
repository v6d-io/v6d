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

// Package annotations contains all annotations provided by vineyard-operator for users
package annotations

const (
	/* following annotations are used for scheduling */

	// VineyardJobRequired is the object ids that required by this job
	VineyardJobRequired = "scheduling.k8s.v6d.io/required"

	/* following annotations are used for operation injection */

	// DaskScheduler is the name of the dask scheduler
	DaskScheduler = "scheduling.k8s.v6d.io/dask-scheduler"
	// DaskWorkerSelector is the selector of the dask worker
	DaskWorkerSelector = "scheduling.k8s.v6d.io/dask-worker-selector"

	/* following annotations are used for sidecar injection */

	// SidecarNameAnno is the annotation key for the sidecar name
	SidecarNameAnno = "sidecar.v6d.io/name"
)
