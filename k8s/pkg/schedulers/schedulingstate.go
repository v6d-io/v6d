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

// Package schedulers implements the vineyard scheduler plugin.
package schedulers

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SchedulerState records the status of current scheduling
type SchedulerState struct {
	client.Client
	state map[string]map[string]string // { jobname: { pod: nodename }}
}

// Append records the action of appending a new pod in job to given node.
func (ss *SchedulerState) Append(job string, pod string, nodeName string) error {
	klog.V(5).Infof("assign job %v pod %v to node %v", job, pod, nodeName)
	if s, ok := ss.state[job]; ok {
		if _, ok := s[pod]; ok {
			return fmt.Errorf("The pod has already been scheduled")
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
// then create the relavant operation CRD.
// 3. Use Best-effort scheduling strategy to schedule the job.
func (ss *SchedulerState) Compute(ctx context.Context, job string, replica int64, rank int64,
	workernodes []string, requires []string, nodeName string, pod *v1.Pod) (int64, error) {
	// if requires no vineyard object, the job can be deployed in any nodes.
	// use round-robin scheduling here
	num := len(workernodes)
	if len(requires) == 0 {
		if workernodes[int(rank)%num] == nodeName {
			klog.V(5).Infof("nodeName: %v, workernodes: %v, rank: %v", nodeName, workernodes, rank)
			return 100, nil
		}
		return 1, nil
	}
	// if no replica, raise
	if replica == 0 {
		return 0, fmt.Errorf("No replica information in the job spec")
	}

	// accumulates all local required objects
	globalObjects, err := ss.getGlobalObjectsByID(ctx, requires)
	if err != nil {
		return 0, err
	}
	klog.V(5).Infof("job %v requires objects %v", job, globalObjects)
	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := ss.getLocalObjectsBySignatures(ctx, localsigs)
	if err != nil {
		return 0, err
	}
	if len(localObjects) == 0 && len(globalObjects) != 0 {
		return 0, fmt.Errorf("No local chunks found")
	}

	if err := ss.createConfigmapForID(ctx, requires, pod.GetNamespace(), localObjects, globalObjects, pod); err != nil {
		klog.V(5).Infof("can't create configmap for object ID %v", err)
	}

	s, err := ss.checkOperationLabels(ctx, pod)
	if err != nil {
		return 0, err
	}
	if s == 0 {
		return 0, nil
	}

	klog.V(5).Infof("job %v requires local chunks %v", job, localObjects)

	locations := make(map[string][]string)
	for _, localObject := range localObjects {
		host := localObject.Spec.Hostname
		if _, ok := locations[host]; !ok {
			locations[host] = make([]string, 0)
		}
		locations[host] = append(locations[host], localObject.Spec.ObjectID)
	}

	// total frags
	totalfrags := int64(len(localObjects))
	// frags for per pod
	nchunks := totalfrags / replica
	if totalfrags%replica != 0 {
		nchunks++
	}

	// find the node
	nodes := make([]string, 0)
	for k := range locations {
		nodes = append(nodes, k)
	}
	sort.Strings(nodes)

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
			klog.V(5).Infof("Bint the pod to the node with the most locations, %v", nodes[0])
			return 100, nil
		}
		return 1, nil
	} else if target == nodeName {
		klog.V(5).Infof("target == nodeName")
		return 100, nil
	} else {
		return 1, nil
	}
}

func (ss *SchedulerState) checkOperationLabels(ctx context.Context, pod *v1.Pod) (int64, error) {
	operationLabels := []string{"assembly.v6d.io/enabled", "repartition.v6d.io/enabled"}
	for _, label := range operationLabels {
		if value, ok := pod.Labels[label]; ok && strings.ToLower(value) == "true" {
			opName := label[:strings.Index(label, ".")]
			op := &v1alpha1.Operation{}
			err := ss.Get(ctx, types.NamespacedName{Name: pod.Name, Namespace: pod.Namespace}, op)
			if err != nil && !apierrors.IsNotFound(err) {
				return 0, err
			}
			if apierrors.IsNotFound(err) {
				requiredJob := pod.Annotations[annotations.VineyardJobRequired]
				targetJob := pod.Labels[labels.VineyardJobName]
				operation := &v1alpha1.Operation{
					ObjectMeta: metav1.ObjectMeta{
						Name:      pod.Name,
						Namespace: pod.Namespace,
					},
					Spec: v1alpha1.OperationSpec{
						Name:           opName,
						Type:           pod.Labels[opName+".v6d.io/type"],
						Require:        requiredJob,
						Target:         targetJob,
						TimeoutSeconds: 300,
					},
				}
				if err := ss.Create(ctx, operation); err != nil {
					return 0, err
				}
			}
			if op.Status.State != v1alpha1.OperationSucceeded {
				return 0, fmt.Errorf("operation %v is not succeeded, state is: %v", opName, op.Status.State)
			}
		}
	}
	return 1, nil
}

// getGlobalObjectsByID returns the global objects by the given jobname.
func (ss *SchedulerState) getGlobalObjectsByID(ctx context.Context, jobNames []string) ([]*v1alpha1.GlobalObject, error) {
	requiredJobs := make(map[string]bool)
	for _, n := range jobNames {
		requiredJobs[n] = true
	}
	objects := []*v1alpha1.GlobalObject{}
	globalObjects := &v1alpha1.GlobalObjectList{}
	if err := ss.List(ctx, globalObjects); err != nil {
		klog.V(5).Infof("client.List failed to get global objects, error: %v", err)
		return nil, err
	}
	for _, obj := range globalObjects.Items {
		if jobname, exist := obj.Labels["k8s.v6d.io/job"]; exist && requiredJobs[jobname] {
			objects = append(objects, &obj)
		}
	}

	return objects, nil
}

// getLocalObjectsBySignatures get all local objects by global objects' signatures
func (ss *SchedulerState) getLocalObjectsBySignatures(ctx context.Context, signatures []string) ([]*v1alpha1.LocalObject, error) {
	objects := make([]*v1alpha1.LocalObject, 0)
	for _, sig := range signatures {
		localObjects := &v1alpha1.LocalObjectList{}
		if err := ss.List(ctx, localObjects, client.MatchingLabels{
			"k8s.v6d.io/signature": sig,
		}); err != nil {
			klog.V(5).Infof("client.List failed to get local objects, error: %v", err)
			return nil, err
		}
		for _, localObject := range localObjects.Items {
			objects = append(objects, &localObject)
		}
	}

	return objects, nil
}

// Create a configmap for the object id and the nodes
func (ss *SchedulerState) createConfigmapForID(ctx context.Context, jobname []string, namespace string,
	localobjects []*v1alpha1.LocalObject, globalobjects []*v1alpha1.GlobalObject, pod *v1.Pod) error {
	for i := range jobname {
		configmap := &v1.ConfigMap{}
		err := ss.Client.Get(ctx, client.ObjectKey{Namespace: namespace, Name: jobname[i]}, configmap)
		if err != nil && !apierrors.IsNotFound(err) {
			klog.V(5).Infof("get configmap error: %v", err)
			return err
		}
		// the configmap doesn't exist
		if apierrors.IsNotFound(err) {
			data := make(map[string]string)
			// get all local objects produced by the required job
			// hostname -> localobject id
			// TODO: if there are lots of localobjects in the same node
			for _, o := range localobjects {
				if (*o).Labels["k8s.v6d.io/job"] == jobname[i] {
					data[(*o).Spec.Hostname] = (*o).Spec.ObjectID
				}
			}
			// get all global objects produced by the required job
			// jobname -> globalobject id
			// TODO: if there are lots of globalobjects produced by the same job
			for _, o := range globalobjects {
				if (*o).Labels["k8s.v6d.io/job"] == jobname[i] {
					data[jobname[i]] = (*o).Spec.ObjectID
				}
			}
			cm := v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobname[i],
					Namespace: namespace,
				},
				Data: data,
			}
			cm.OwnerReferences = pod.GetOwnerReferences()
			if err := ss.Client.Create(ctx, &cm); err != nil {
				klog.V(5).Infof("create configmap error: %v", err)
				return err
			}
		}
		klog.V(5).Infof("the configmap [%v/%v] exist!", namespace, jobname[i])
	}

	return nil
}
