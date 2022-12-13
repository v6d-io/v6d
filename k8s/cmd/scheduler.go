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

/* Package main is used for simplify the operator usage. */
package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/schedulers"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var slog = log.Logger.WithName("vineyard-scheduler-outside-cluster")

// Scheduling is used to schedule jobs to nodes
func Scheduling(c client.Client, a, l map[string]string, replica int, namespace string,
	ownerReferences []metav1.OwnerReference) string {
	scheduledOrder := ""
	jobToNode := make(map[string]int)
	// get all nodes that have vineyardd
	vineyarddName := l[labels.VineyarddName]
	vineyarddNamespace := l[labels.VineyarddNamespace]
	allNodes := schedulers.GetVineyarddNodes(c, slog, vineyarddName, vineyarddNamespace)

	// use round-robin to schedule workload here
	if a["scheduling.k8s.v6d.io/required"] == "none" {
		l := len(allNodes)
		for i := 0; i < replica; i++ {
			jobToNode[allNodes[i%l]]++
		}

		s := make([]string, 0)
		for n, v := range jobToNode {
			s = append(s, n+"="+strconv.Itoa(v))
		}
		scheduledOrder = strings.Join(s, ",")
		return scheduledOrder
	}

	// get required jobs
	required, err := schedulers.GetRequiredJob(slog, a)
	if err != nil {
		slog.Info(fmt.Sprintf("get required jobs failed: %v", err))
		return ""
	}
	// get all global objects
	globalObjects, err := schedulers.GetGlobalObjectsByID(c, slog, required)
	if err != nil {
		slog.Info(fmt.Sprintf("get global objects failed: %v", err))
		return ""
	}

	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := schedulers.GetLocalObjectsBySignatures(c, slog, localsigs)
	if err != nil {
		slog.Info(fmt.Sprintf("get local objects failed: %v", err))
		return ""
	}

	locations, nchunks, nodes := schedulers.GetObjectInfo(localObjects, int64(replica))

	var cnt int64

	for i := 0; i < replica; i++ {
		rank := int64(i)
		for _, node := range nodes {
			localfrags := int64(len(locations[node]))
			if cnt+localfrags >= (nchunks*rank + (nchunks+1)/2) {
				jobToNode[node]++
				break
			}
			cnt += localfrags
		}
	}

	fmt.Println("nodes: ", nodes)
	fmt.Println("locations: ", locations)
	fmt.Println("replica: ", replica)
	fmt.Println("jobToNode: ", jobToNode)

	s := make([]string, 0)
	for n, v := range jobToNode {
		s = append(s, n+"="+strconv.Itoa(v))
	}
	scheduledOrder = strings.Join(s, ",")

	if err := schedulers.CreateConfigmapForID(c, slog, required, namespace, localObjects, globalObjects, ownerReferences); err != nil {
		slog.Info(fmt.Sprintf("can't create configmap for object ID %v", err))
	}

	return scheduledOrder
}
