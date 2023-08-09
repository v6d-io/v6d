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
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// VineyardSchedulerOutsideCluster is the vineyard scheduler outside cluster
type VineyardSchedulerOutsideCluster struct {
	client.Client
	annotations map[string]string
	labels      map[string]string
	config      SchedulerConfig
	withoutCRD  bool
}

// NewVineyardSchedulerOutsideCluster returns a new vineyard scheduler outside cluster
func NewVineyardSchedulerOutsideCluster(
	c client.Client,
	annotations map[string]string,
	labels map[string]string,
	namespace string,
	ownerReferences []metav1.OwnerReference,
) *VineyardSchedulerOutsideCluster {
	return &VineyardSchedulerOutsideCluster{
		Client:      c,
		annotations: annotations,
		labels:      labels,
		config: SchedulerConfig{
			Namespace:      namespace,
			OwnerReference: &ownerReferences,
		},
	}
}

func (vs *VineyardSchedulerOutsideCluster) SetWithoutCRD(withoutCRD bool) *VineyardSchedulerOutsideCluster {
	vs.withoutCRD = withoutCRD
	return vs
}

// SetupConfig setups the scheduler config
func (vs *VineyardSchedulerOutsideCluster) SetupConfig() error {
	required := GetRequiredJob(vs.annotations)

	vs.config.Required = required

	nodes, err := GetVineyarddNodes(vs.Client, vs.labels)
	if err != nil {
		return err
	}
	vs.config.Nodes = nodes

	return nil
}

// buildSchedulerOrder builds the scheduler order from the job to node map
func (vs *VineyardSchedulerOutsideCluster) buildSchedulerOrder(jobToNode map[string]int) string {
	scheduledOrder := ""
	s := make([]string, 0)
	for n, v := range jobToNode {
		s = append(s, n+"="+strconv.Itoa(v))
	}
	scheduledOrder = strings.Join(s, ",")
	return scheduledOrder
}

// Schedule reads the replica of workload and returns the scheduler order
func (vs *VineyardSchedulerOutsideCluster) Schedule(replica int) (string, error) {
	jobToNode := make(map[string]int)
	// if there are no required jobs, use round robin strategy
	roundRobin := NewRoundRobinStrategy(vs.config.Nodes)

	if len(vs.config.Required) == 0 {
		for i := 0; i < replica; i++ {
			node, err := roundRobin.Compute(i)
			if err != nil {
				return "", err
			}
			jobToNode[node]++
		}
		return vs.buildSchedulerOrder(jobToNode), nil
	}

	// if there are required jobs, use best effort strategy
	bestEffort := NewBestEffortStrategy(
		vs.Client,
		vs.config.Required,
		replica,
		vs.config.Namespace,
		vs.config.OwnerReference,
	)

	deploymentName := vs.labels[labels.VineyarddName]
	namespace := vs.labels[labels.VineyarddNamespace]
	var node string
	var err error
	for i := 0; i < replica; i++ {
		if vs.withoutCRD {
			node, err = bestEffort.TrackingChunksByAPI(deploymentName, namespace).Compute(i)
		} else {
			node, err = bestEffort.TrackingChunksByCRD().Compute(i)
		}
		if err != nil {
			return "", err
		}
		if node != "" {
			jobToNode[node]++
		}
	}

	return vs.buildSchedulerOrder(jobToNode), nil
}
