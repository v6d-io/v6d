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
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
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

// VineyardSchedulerOnKubernetes is the vineyard scheduler on kubernetes
type VineyardSchedulerOnKubernetes struct {
	client.Client
	pod     *v1.Pod
	rank    int
	replica int
	config  SchedulerConfig
}

// NewVineyardSchedulerOnKubernetes returns a new vineyard scheduler on kubernetes
func NewVineyardSchedulerOnKubernetes(
	c client.Client,
	pod *v1.Pod,
	rank int,
	replica int,
) *VineyardSchedulerOnKubernetes {
	return &VineyardSchedulerOnKubernetes{
		Client:  c,
		pod:     pod,
		rank:    rank,
		replica: replica,
	}
}

// SetupConfig setups the scheduler config
func (vsok *VineyardSchedulerOnKubernetes) SetupConfig() error {
	pod := vsok.pod

	required := GetRequiredJob(pod.Annotations)

	vsok.config.Required = required

	nodes, err := GetVineyarddNodes(vsok.Client, pod.Labels)
	if err != nil {
		return err
	}
	vsok.config.Nodes = nodes

	vsok.config.Namespace = pod.Namespace
	vsok.config.OwnerReference = &pod.OwnerReferences
	return nil
}

// CheckOperationLabels checks the operation labels and creates the operation if necessary
func (vsok *VineyardSchedulerOnKubernetes) CheckOperationLabels() (int, error) {
	pod := vsok.pod
	operationLabels := []string{"assembly.v6d.io/enabled", "repartition.v6d.io/enabled"}
	for _, label := range operationLabels {
		if value, ok := pod.Labels[label]; ok && strings.ToLower(value) == "true" {
			opName := label[:strings.Index(label, ".")]
			op := &v1alpha1.Operation{}
			err := vsok.Get(
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
				if err := vsok.Create(context.TODO(), operation); err != nil {
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
func (vsok *VineyardSchedulerOnKubernetes) Schedule(nodeName string) (int, error) {
	// if there are no required jobs, use round robin strategy
	roundRobin := NewRoundRobinStrategy(vsok.config.Nodes)

	if len(vsok.config.Required) == 0 {
		target, _ := roundRobin.Compute(vsok.rank)
		if target == nodeName {
			return 100, nil
		}
		return 1, nil
	}

	// if there are required jobs, use best effort strategy
	bestEffort := NewBestEffortStrategy(
		vsok.Client,
		vsok.config.Required,
		vsok.replica,
		vsok.config.Namespace,
		vsok.config.OwnerReference,
	)

	target, err := bestEffort.Compute(vsok.rank)
	if err != nil {
		return 0, err
	}

	s, err := vsok.CheckOperationLabels()
	if err != nil {
		return 0, err
	}
	if s == 0 {
		return 0, nil
	}

	// make sure every pod will be deployed in a node
	if target == "" {
		if nodeName == vsok.config.Nodes[0] {
			return 100, nil
		}
		return 1, nil
	} else if target == nodeName {
		return 100, nil
	} else {
		return 1, nil
	}
}
