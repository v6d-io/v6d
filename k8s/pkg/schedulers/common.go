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
	"sort"
	"strings"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

// GetVineyarddNodes returns all node names of vineyardd pods.
func GetVineyarddNodes(c client.Client, log log.Logger, name, namespace string) []string {
	nodes := []string{}

	podList := v1.PodList{}
	option := &client.ListOptions{
		LabelSelector: apilabels.SelectorFromSet(apilabels.Set{
			"app.kubernetes.io/name":     name,
			"app.kubernetes.io/instance": namespace + "-" + name,
		}),
	}
	if namespace != "" {
		option.Namespace = namespace
	}
	if err := c.List(context.TODO(), &podList, option); err != nil {
		log.Info("Failed to list all pods with the specific label", "error", err)
	}

	for _, pod := range podList.Items {
		nodes = append(nodes, pod.Spec.NodeName)
	}
	sort.Strings(nodes)
	return nodes
}

// GetRequiredJob get all required jobs name that separated by '.' from annotations
func GetRequiredJob(log log.Logger, anno map[string]string) ([]string, error) {
	objects, exists := anno[annotations.VineyardJobRequired]
	if !exists {
		return []string{}, errors.Errorf(
			"Failed to get the required jobs, please set none if there is no required job",
		)
	}

	log.Info(fmt.Sprintf("Get the required jobs: %v", objects))
	if objects == "none" {
		return []string{}, nil
	}
	return strings.Split(objects, "."), nil
}

// GetLocalObjectsBySignatures returns the local objects by the given signatures.
func GetLocalObjectsBySignatures(
	c client.Client,
	log log.Logger,
	signatures []string,
) ([]*v1alpha1.LocalObject, error) {
	objects := make([]*v1alpha1.LocalObject, 0)
	for _, sig := range signatures {
		localObjects := &v1alpha1.LocalObjectList{}
		if err := c.List(context.TODO(), localObjects, client.MatchingLabels{
			"k8s.v6d.io/signature": sig,
		}); err != nil {
			log.Info(fmt.Sprintf("client.List failed to get local objects, error: %v", err))
			return nil, err
		}
		for _, localObject := range localObjects.Items {
			objects = append(objects, &localObject)
		}
	}

	return objects, nil
}

// GetGlobalObjectsByID returns the global objects by the given jobname.
func GetGlobalObjectsByID(
	c client.Client,
	log log.Logger,
	jobNames []string,
) ([]*v1alpha1.GlobalObject, error) {
	requiredJobs := make(map[string]bool)
	for _, n := range jobNames {
		requiredJobs[n] = true
	}
	objects := []*v1alpha1.GlobalObject{}
	globalObjects := &v1alpha1.GlobalObjectList{}
	if err := c.List(context.TODO(), globalObjects); err != nil {
		log.Info(fmt.Sprintf("client.List failed to get global objects, error: %v", err))
		return nil, err
	}
	for i, obj := range globalObjects.Items {
		if jobname, exist := obj.Labels["k8s.v6d.io/job"]; exist && requiredJobs[jobname] {
			objects = append(objects, &globalObjects.Items[i])
		}
	}

	return objects, nil
}

// CheckOperationLabels checks if the pod has operation labels and returns the operation name.
func CheckOperationLabels(c client.Client, log log.Logger, pod *v1.Pod) (int64, error) {
	operationLabels := []string{"assembly.v6d.io/enabled", "repartition.v6d.io/enabled"}
	for _, label := range operationLabels {
		if value, ok := pod.Labels[label]; ok && strings.ToLower(value) == "true" {
			opName := label[:strings.Index(label, ".")]
			op := &v1alpha1.Operation{}
			err := c.Get(
				context.TODO(),
				types.NamespacedName{Name: pod.Name, Namespace: pod.Namespace},
				op,
			)
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
				if err := c.Create(context.TODO(), operation); err != nil {
					return 0, err
				}
			}
			if op.Status.State != v1alpha1.OperationSucceeded {
				return 0, errors.Errorf(
					"operation %v is not succeeded, state is: %v",
					opName,
					op.Status.State,
				)
			}
		}
	}
	return 1, nil
}

// CreateConfigmapForID creates a configmap for the object id and the nodes.
func CreateConfigmapForID(
	c client.Client,
	log log.Logger,
	jobname []string,
	namespace string,
	localobjects []*v1alpha1.LocalObject,
	globalobjects []*v1alpha1.GlobalObject,
	ownerReference []metav1.OwnerReference,
) error {
	for i := range jobname {
		configmap := &v1.ConfigMap{}
		err := c.Get(
			context.TODO(),
			client.ObjectKey{Namespace: namespace, Name: jobname[i]},
			configmap,
		)
		if err != nil && !apierrors.IsNotFound(err) {
			log.Info(fmt.Sprintf("get configmap error:: %v", err))
			return err
		}
		// the configmap doesn't exist
		if apierrors.IsNotFound(err) {
			data := make(map[string]string)
			localObjList := make(map[string][]string)
			// get all local objects produced by the required job
			// hostname -> localobject id
			for _, o := range localobjects {
				if (*o).Labels["k8s.v6d.io/job"] == jobname[i] {
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
				if (*o).Labels["k8s.v6d.io/job"] == jobname[i] {
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
					Namespace: namespace,
				},
				Data: data,
			}
			cm.OwnerReferences = ownerReference
			if err := c.Create(context.TODO(), &cm); err != nil {
				log.Error(err, "create configmap error")
				return err
			}
		}
		log.Info(fmt.Sprintf("the configmap [%v/%v] exist!", namespace, jobname[i]))
	}

	return nil
}

// GetObjectInfo returns the local object info including the locations and average number of chunks per node.
func GetObjectInfo(
	localObjects []*v1alpha1.LocalObject,
	replica int64,
) (map[string][]string, int64, []string) {
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
	return locations, nchunks, nodes
}
