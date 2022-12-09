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

package main

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func Scheduling(c client.Client, a, l map[string]string, replica int, namespace string) string {
	scheduledOrder := ""
	ctx := context.Background()
	jobToNode := make(map[string]int)
	// get all nodes that have vineyardd
	nodes := []string{}

	vineyarddName := l[labels.VineyarddName]
	vineyarddNamespace := l[labels.VineyarddNamespace]

	podList := v1.PodList{}
	option := &client.ListOptions{
		LabelSelector: apilabels.SelectorFromSet(apilabels.Set{
			"app.kubernetes.io/name":     vineyarddName,
			"app.kubernetes.io/instance": "vineyardd",
		}),
		Namespace: vineyarddNamespace,
	}
	if err := c.List(ctx, &podList, option); err != nil {
		fmt.Println("Failed to list all pods with the specific label: %v", err)
	}

	for _, pod := range podList.Items {
		nodes = append(nodes, pod.Spec.NodeName)
	}

	if a["scheduling.k8s.v6d.io/required"] == "none" {
		l := len(nodes)
		for i := 0; i < replica; i++ {
			jobToNode[nodes[i%l]]++
		}

		s := make([]string, 0)
		for n, v := range jobToNode {
			s = append(s, n+"="+strconv.Itoa(v))
		}
		scheduledOrder = strings.Join(s, ",")
		return scheduledOrder
	}

	// get required jobs
	required := []string{}
	jobs, exists := a[annotations.VineyardJobRequired]
	if !exists {
		fmt.Println("Failed to get the required jobs, please set none if there is no required job")
	}
	fmt.Println(" jobs: ", jobs)
	required = strings.Split(jobs, ".")
	fmt.Println("required jobs: ", required)
	// get all global objects
	requiredJobs := make(map[string]bool)
	for _, n := range required {
		requiredJobs[n] = true
	}
	objects := []*v1alpha1.GlobalObject{}
	globalObjects := &v1alpha1.GlobalObjectList{}
	if err := c.List(ctx, globalObjects); err != nil {
		fmt.Println("client.List failed to get global objects, error: %v", err)
	}
	for i := range globalObjects.Items {
		if jobname, exist := globalObjects.Items[i].Labels["k8s.v6d.io/job"]; exist && requiredJobs[jobname] {
			objects = append(objects, &globalObjects.Items[i])
			fmt.Println("global objects name: ", globalObjects.Items[i].Name)
		}
	}

	fmt.Println("all global objects: ", objects)
	localsigs := make([]string, 0)
	for _, globalObject := range objects {
		fmt.Println("all global objects: ", globalObject.Name)
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}

	lobjects := make([]*v1alpha1.LocalObject, 0)
	for _, sig := range localsigs {
		localObjects := &v1alpha1.LocalObjectList{}
		if err := c.List(ctx, localObjects, client.MatchingLabels{
			"k8s.v6d.io/signature": sig,
		}); err != nil {
			fmt.Println("client.List failed to get local objects, error: %v", err)
		}
		for _, localObject := range localObjects.Items {
			lobjects = append(lobjects, &localObject)
		}
	}

	locations := make(map[string][]string)
	for _, localObject := range lobjects {
		host := localObject.Spec.Hostname
		if _, ok := locations[host]; !ok {
			locations[host] = make([]string, 0)
		}
		locations[host] = append(locations[host], localObject.Spec.ObjectID)
	}
	fmt.Println("locations: ", locations)

	// total frags
	totalfrags := int(len(lobjects))
	// frags for per pod
	nchunks := totalfrags / replica
	if totalfrags%replica != 0 {
		nchunks++
	}

	// find the node
	objNodes := make([]string, 0)
	for k := range locations {
		objNodes = append(objNodes, k)
	}
	sort.Strings(objNodes)

	var cnt int
	//jobToNode := make(map[string]int)

	for i := 0; i < replica; i++ {
		rank := i
		for _, node := range nodes {
			localfrags := len(locations[node])
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

	for i := range required {
		configmap := &v1.ConfigMap{}
		err := c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: required[i]}, configmap)
		if err != nil && !apierrors.IsNotFound(err) {
			fmt.Println("client.Get failed to get configmap, error: %v", err)
		}
		// the configmap doesn't exist
		if apierrors.IsNotFound(err) {
			data := make(map[string]string)
			// get all local objects produced by the required job
			// hostname -> localobject id
			// TODO: if there are lots of localobjects in the same node
			for _, o := range lobjects {
				if (*o).Labels["k8s.v6d.io/job"] == required[i] {
					data[(*o).Spec.Hostname] = (*o).Spec.ObjectID
				}
			}
			// get all global objects produced by the required job
			// jobname -> globalobject id
			// TODO: if there are lots of globalobjects produced by the same job
			for _, o := range objects {
				if (*o).Labels["k8s.v6d.io/job"] == required[i] {
					data[required[i]] = (*o).Spec.ObjectID
				}
			}
			cm := v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      required[i],
					Namespace: namespace,
				},
				Data: data,
			}
			if err := c.Create(ctx, &cm); err != nil {
				fmt.Println("create configmap error: %v", err)
			}
		}
	}

	return scheduledOrder
}
