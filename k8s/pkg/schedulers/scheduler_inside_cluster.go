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
	"strings"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

// Scheduler is the interface for all vineyard schedulers
type Scheduler interface {
	Schedule(interface{}) (interface{}, error)
}

// SchedulerConfig is the common configuration for all schedulers
type SchedulerConfig struct {
	Required       []string
	Nodes          []string
	Namespace      string
	OwnerReference *[]metav1.OwnerReference
}

// VineyardSchedulerInsideCluster is the vineyard scheduler inside cluster
type VineyardSchedulerInsideCluster struct {
	client.Client
	pod     *v1.Pod
	rank    int
	replica int
	config  SchedulerConfig
}

// NewVineyardSchedulerInsideCluster returns a new vineyard scheduler inside cluster
func NewVineyardSchedulerInsideCluster(
	c client.Client,
	pod *v1.Pod,
	rank int,
	replica int,
) *VineyardSchedulerInsideCluster {
	return &VineyardSchedulerInsideCluster{
		Client:  c,
		pod:     pod,
		rank:    rank,
		replica: replica,
	}
}

// SetupConfig setups the scheduler config
func (vs *VineyardSchedulerInsideCluster) SetupConfig() error {
	pod := vs.pod

	required := GetRequiredJob(pod.Annotations)

	vs.config.Required = required

	nodes, err := GetVineyarddNodes(vs.Client, pod.Labels)
	if err != nil {
		return err
	}
	vs.config.Nodes = nodes

	vs.config.Namespace = pod.Namespace
	vs.config.OwnerReference = &pod.OwnerReferences
	return nil
}

// checkOperationLabels checks the operation labels and creates the operation if necessary
func (vs *VineyardSchedulerInsideCluster) checkOperationLabels() (int, error) {
	pod := vs.pod
	operationLabels := []string{labels.AssemblyEnabledLabel, labels.RepartitionEnabledLabel}
	for _, label := range operationLabels {
		if value, ok := pod.Labels[label]; ok && strings.ToLower(value) == "true" {
			opName := label[:strings.Index(label, ".")]
			op := &v1alpha1.Operation{}
			err := vs.Get(
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
				if err := vs.Create(context.TODO(), operation); err != nil {
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

// Schedule compute the score for the given node
func (vs *VineyardSchedulerInsideCluster) Schedule(nodeName string) (int, error) {
	// if there are no required jobs, use round robin strategy
	roundRobin := NewRoundRobinStrategy(vs.config.Nodes)

	if len(vs.config.Required) == 0 {
		log.Infof("Start scheduling with round robin strategy...")
		target, err := roundRobin.Compute(vs.rank)
		if err != nil {
			return 1, err
		}
		if target == nodeName {
			return 100, nil
		}
		return 1, nil
	}

	log.Infof("Start scheduling with best effort strategy...")
	// if there are required jobs, use best effort strategy
	bestEffort := NewBestEffortStrategy(
		vs.Client,
		vs.config.Required,
		vs.replica,
		vs.config.Namespace,
		vs.config.OwnerReference,
	)

	target, err := bestEffort.TrackingChunksByCRD().Compute(vs.rank)
	if err != nil {
		return 0, err
	}

	s, err := vs.checkOperationLabels()
	if err != nil {
		return 0, err
	}
	if s == 0 {
		return 0, nil
	}

	// make sure every pod will be deployed in a node
	if target == "" {
		if nodeName == bestEffort.nodes[0] {
			log.Infof("No available node, choose the first node %s to deploy the pod", nodeName)
			return 100, nil
		}
		return 1, nil
	} else if target == nodeName {
		return 100, nil
	} else {
		return 1, nil
	}
}
