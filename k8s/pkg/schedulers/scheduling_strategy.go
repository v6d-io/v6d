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
	"sort"
	"strings"

	"github.com/pkg/errors"
	"go.uber.org/multierr"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
)

// SchedulerStrategy is the interface for all scheduler strategies.
type SchedulerStrategy interface {
	// Compute returns the score of each node.
	Compute(interface{}) (interface{}, error)
}

// RoundRobinStrategy is the round robin strategy.
type RoundRobinStrategy struct {
	nodes []string
}

// NewRoundRobinStrategy returns a new RoundRobinStrategy.
func NewRoundRobinStrategy(nodes []string) *RoundRobinStrategy {
	return &RoundRobinStrategy{
		nodes: nodes,
	}
}

// Compute returns the node by the given rank.
func (r *RoundRobinStrategy) Compute(rank int) (string, error) {
	l := len(r.nodes)
	return r.nodes[rank%l], nil
}

// BestEffortStrategy is the best effort strategy.
type BestEffortStrategy struct {
	client.Client
	// the required jobs
	required []string
	// the replicas of the job(pod or workload)
	replica int
	// the namespace of the job(pod or workload)
	namespace string
	// the ownerReference of created configmap
	ownerReference *[]metav1.OwnerReference
}

// NewBestEffortStrategy returns a new BestEffortStrategy.
func NewBestEffortStrategy(
	client client.Client,
	required []string,
	replica int,
	namespace string,
	ownerReference *[]metav1.OwnerReference,
) *BestEffortStrategy {
	return &BestEffortStrategy{
		Client:         client,
		required:       required,
		replica:        replica,
		namespace:      namespace,
		ownerReference: ownerReference,
	}
}

// GetLocalObjectsBySignatures returns the local objects by the given signatures.
func (b *BestEffortStrategy) GetLocalObjectsBySignatures(
	signatures []string,
) ([]*v1alpha1.LocalObject, error) {
	objects := make([]*v1alpha1.LocalObject, 0)
	for _, sig := range signatures {
		localObjects := &v1alpha1.LocalObjectList{}
		if err := b.List(context.TODO(), localObjects, client.MatchingLabels{
			"k8s.v6d.io/signature": sig,
		}); err != nil {
			return nil, err
		}
		for _, localObject := range localObjects.Items {
			objects = append(objects, &localObject)
		}
	}

	return objects, nil
}

// GetObjectInfo returns the local object info including the locations and average number of chunks per node.
func (b *BestEffortStrategy) GetObjectInfo(
	localObjects []*v1alpha1.LocalObject,
	replica int,
) (map[string][]string, int, []string) {
	locations := make(map[string][]string)
	for _, localObject := range localObjects {
		host := localObject.Spec.Hostname
		if _, ok := locations[host]; !ok {
			locations[host] = make([]string, 0)
		}
		locations[host] = append(locations[host], localObject.Spec.ObjectID)
	}

	// total frags
	totalfrags := len(localObjects)
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
	return locations, nchunks, nodes
}

// GetGlobalObjectsByID returns the global objects by the given jobname.
func (b *BestEffortStrategy) GetGlobalObjectsByID(
	jobNames []string,
) ([]*v1alpha1.GlobalObject, error) {
	requiredJobs := make(map[string]bool)
	for _, n := range jobNames {
		requiredJobs[n] = true
	}
	objects := []*v1alpha1.GlobalObject{}
	globalObjects := &v1alpha1.GlobalObjectList{}
	if err := b.List(context.TODO(), globalObjects); err != nil {
		return nil, err
	}
	for i, obj := range globalObjects.Items {
		if jobname, exist := obj.Labels[labels.VineyardObjectJobLabel]; exist && requiredJobs[jobname] {
			objects = append(objects, &globalObjects.Items[i])
		}
	}

	return objects, nil
}

// CreateConfigmapForID creates a configmap for the object id and the nodes.
func (b *BestEffortStrategy) CreateConfigmapForID(
	jobname []string,
	localobjects []*v1alpha1.LocalObject,
	globalobjects []*v1alpha1.GlobalObject,
) error {
	for i := range jobname {
		configmap := &v1.ConfigMap{}
		err := b.Get(
			context.TODO(),
			client.ObjectKey{Namespace: b.namespace, Name: jobname[i]},
			configmap,
		)
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		// the configmap doesn't exist
		if apierrors.IsNotFound(err) {
			data := make(map[string]string)
			localObjList := make(map[string][]string)
			// get all local objects produced by the required job
			// hostname -> localobject id
			for _, o := range localobjects {
				if (*o).Labels[labels.VineyardObjectJobLabel] == jobname[i] {
					localObjList[(*o).Spec.Hostname] = append(
						localObjList[(*o).Spec.Hostname],
						(*o).Spec.ObjectID,
					)
				}
			}
			for nodeName, nodeObjs := range localObjList {
				data[nodeName] = strings.Join(nodeObjs, ",")
			}
			// get all global objects produced by the required job
			// jobname -> globalobject id
			globalObjs := []string{}
			for _, o := range globalobjects {
				if (*o).Labels[labels.VineyardObjectJobLabel] == jobname[i] {
					globalObjs = append(globalObjs, (*o).Spec.ObjectID)
				}
			}
			data[jobname[i]] = strings.Join(globalObjs, ",")
			cm := v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobname[i],
					Namespace: b.namespace,
				},
				Data: data,
			}
			cm.OwnerReferences = *b.ownerReference
			if err := b.Create(context.TODO(), &cm); err != nil {
				return err
			}
		}
	}

	return nil
}

// Compute return the target node for the given rank.
func (b *BestEffortStrategy) Compute(rank int) (string, error) {
	var errList error
	target := ""

	// accumulates all local required objects
	globalObjects, err := b.GetGlobalObjectsByID(b.required)
	if err != nil {
		_ = multierr.Append(errList, err)
	}

	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := b.GetLocalObjectsBySignatures(localsigs)
	if err != nil {
		_ = multierr.Append(errList, err)
	}

	// if there is no local objects, return error
	if len(localObjects) == 0 && len(globalObjects) != 0 {
		_ = multierr.Append(errList, errors.Errorf("No local chunks found"))
		return target, errList
	}

	locations, nchunks, nodes := b.GetObjectInfo(localObjects, b.replica)

	cnt := 0
	for _, node := range nodes {
		localfrags := len(locations[node])
		if cnt+localfrags >= (nchunks*rank + (nchunks+1)/2) {
			target = node
			break
		}
		cnt += localfrags
	}

	// create configmap for each job
	if err := b.CreateConfigmapForID(b.required, localObjects, globalObjects); err != nil {
		_ = multierr.Append(errList, err)
	}

	return target, errList
}
