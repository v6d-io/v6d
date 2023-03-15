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

// Package schedulers implements the vineyard scheduler plugin.
package schedulers

import (
	"context"
	"fmt"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SchedulerState records the status of current scheduling
type SchedulerState struct {
	client.Client
	state map[string]map[string]string // { jobname: { pod: nodename }}
}

// Append records the action of appending a new pod in job to given node.
func (ss *SchedulerState) Append(job string, pod string, nodeName string) error {
	slog.Info(fmt.Sprintf("assign job %v pod %v to node %v", job, pod, nodeName))
	if s, ok := ss.state[job]; ok {
		if _, ok := s[pod]; ok {
			return errors.Errorf("The pod has already been scheduled")
		}
		s[pod] = nodeName
		return nil
	}
	ss.state[job] = make(map[string]string)
	ss.state[job][pod] = nodeName
	return nil
}

// Compute will compute the scheduling score for the given job through the following strategies.
// 1. if there is no required job then use round-robin scheduling
// 2. Scan the pod's labels, if there is an operation annotation in pod, such as 'assembly'、'repartition' etc,
// then create the relevant operation CRD.
// 3. Use Best-effort scheduling strategy to schedule the job.
func (ss *SchedulerState) Compute(ctx context.Context, job string, replica int64, rank int64,
	workernodes []string, requires []string, nodeName string, pod *v1.Pod,
) (int64, error) {
	// if requires no vineyard object, the job can be deployed in any nodes.
	// use round-robin scheduling here
	num := len(workernodes)
	if len(requires) == 0 {
		if workernodes[int(rank)%num] == nodeName {
			slog.Info(
				fmt.Sprintf("nodeName: %v, workernodes: %v, rank: %v", nodeName, workernodes, rank),
			)
			return 100, nil
		}
		return 1, nil
	}
	// if no replica, raise
	if replica == 0 {
		return 0, errors.Errorf("No replica information in the job spec")
	}

	// accumulates all local required objects
	globalObjects, err := GetGlobalObjectsByID(ss.Client, slog, requires)
	if err != nil {
		return 0, err
	}
	slog.Info(fmt.Sprintf("job %v requires objects %v", job, globalObjects))
	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := GetLocalObjectsBySignatures(ss.Client, slog, localsigs)
	if err != nil {
		return 0, err
	}
	if len(localObjects) == 0 && len(globalObjects) != 0 {
		return 0, errors.Errorf("No local chunks found")
	}

	ownerReferences := pod.GetOwnerReferences()
	if err := CreateConfigmapForID(ss.Client, slog, requires, pod.GetNamespace(), localObjects, globalObjects, ownerReferences); err != nil {
		slog.Info(fmt.Sprintf("can't create configmap for object ID %v", err))
	}

	s, err := CheckOperationLabels(ss.Client, slog, pod)
	if err != nil {
		return 0, err
	}
	if s == 0 {
		return 0, nil
	}

	slog.Info(fmt.Sprintf("job %v requires local chunks %v", job, localObjects))

	locations, nchunks, nodes := GetObjectInfo(localObjects, replica)

	var cnt int64
	target := ""
	for _, node := range nodes {
		localfrags := int64(len(locations[node]))
		if cnt+localfrags >= (nchunks*rank + (nchunks+1)/2) {
			target = node
			break
		}
		cnt += localfrags
	}

	// make sure every pod will be deployed in a node
	if target == "" {
		if nodeName == nodes[0] {
			slog.Info(fmt.Sprintf("Bint the pod to the node with the most locations, %v", nodes[0]))
			return 100, nil
		}
		return 1, nil
	} else if target == nodeName {
		slog.Info("target == nodeName")
		return 100, nil
	} else {
		return 1, nil
	}
}
