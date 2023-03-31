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
	"strings"
	"time"

	"github.com/pkg/errors"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var slog = log.WithName("vineyard-scheduler-in-cluster")

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = "Vineyard"
	// Timeout is the default timeout for the scheduler plugin.
	Timeout = 60
	// VineyardSystemNamespace is the default system namespace
	VineyardSystemNamespace = "vineyard-system"
)

// VineyardScheduling is a plugin that schedules pods that requires vineyard objects as inputs.
type VineyardScheduling struct {
	client.Client
	handle          framework.Handle
	scheduleTimeout *time.Duration
	podRank         map[string]map[string]int
}

// New initializes a vineyard scheduler
// func New(configuration *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
func New(
	client client.Client,
	config *rest.Config,
	obj runtime.Object,
	handle framework.Handle,
) (framework.Plugin, error) {
	slog.Info("Initializing the vineyard scheduler plugin ...")
	timeout := Timeout * time.Second
	scheduling := &VineyardScheduling{
		Client:          client,
		handle:          handle,
		scheduleTimeout: &timeout,
		podRank:         map[string]map[string]int{},
	}
	return scheduling, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (vs *VineyardScheduling) Name() string {
	return Name
}

// Less compares the priority of two
func (vs *VineyardScheduling) Less(pod1, pod2 *framework.PodInfo) bool {
	prio1 := corev1.PodPriority(pod1.Pod)
	prio2 := corev1.PodPriority(pod2.Pod)
	return prio1 > prio2
}

// Score compute the score for a pod based on the status of required vineyard objects.
func (vs *VineyardScheduling) Score(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) (int64, *framework.Status) {
	slog.Info(fmt.Sprintf("scoring for pod %v on node %v", GetNamespacedName(pod), nodeName))
	replica, err := vs.getJobReplica(pod)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	rank := vs.GetPodRank(pod, replica)
	scheduler := NewVineyardSchedulerOnKubernetes(
		vs.Client, pod, rank, replica,
	)
	err = scheduler.SetupConfig()
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	score, err := scheduler.Schedule(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}
	if score == 0 {
		return int64(score), framework.NewStatus(framework.Unschedulable, "Computed store is zero")
	}
	slog.Info(fmt.Sprintf("score for pod %v on node %v is: %v", pod.Name, nodeName, score))
	return int64(score), framework.NewStatus(framework.Success, "")
}

// ScoreExtensions of the Score plugin.
func (vs *VineyardScheduling) ScoreExtensions() framework.ScoreExtensions {
	return vs
}

// NormalizeScore normalizes the score of all nodes for a pod.
func (vs *VineyardScheduling) NormalizeScore(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	scores framework.NodeScoreList,
) *framework.Status {
	// Find highest and lowest scores.
	return framework.NewStatus(framework.Success, "")
}

// Permit only permit runs on the node that has vineyard installed.
func (vs *VineyardScheduling) Permit(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) (*framework.Status, time.Duration) {
	return framework.NewStatus(framework.Success, ""), 0
}

// PostBind prints the bind info
func (vs *VineyardScheduling) PostBind(
	ctx context.Context,
	_ *framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) {
	slog.Info(fmt.Sprintf("Bind pod %v on node %v", GetNamespacedName(pod), nodeName))
}

func (vs *VineyardScheduling) getJobReplica(pod *v1.Pod) (int, error) {
	// infer from the ownership
	ctx := context.TODO()
	// ctx := context.Background()
	for _, owner := range pod.GetOwnerReferences() {
		name := types.NamespacedName{Namespace: pod.Namespace, Name: owner.Name}
		switch owner.Kind {
		case "ReplicaSet":
			replicaset := &appsv1.ReplicaSet{}
			if err := vs.Get(ctx, name, replicaset); err == nil {
				return int(*replicaset.Spec.Replicas), nil
			}
		case "DaemonSet":
			daemonset := &appsv1.DaemonSet{}
			if err := vs.Get(ctx, name, daemonset); err == nil {
				return daemonset.Spec.Size(), nil
			}
		case "StatefulSet":
			statefulset := &appsv1.StatefulSet{}
			if err := vs.Get(ctx, name, statefulset); err == nil {
				return int(*statefulset.Spec.Replicas), nil
			}
		case "Job":
			job := &batchv1.Job{}
			if err := vs.Get(ctx, name, job); err == nil {
				return int(*job.Spec.Parallelism), nil
			}
		case "CronJob":
			crobjob := &batchv1.CronJob{}
			if err := vs.Get(ctx, name, crobjob); err == nil {
				return crobjob.Spec.Size(), nil
			}
		case "Deployment":
			deployment := &appsv1.Deployment{}
			if err := vs.Get(ctx, name, deployment); err == nil {
				return int(*deployment.Spec.Replicas), nil
			}
		default:
			slog.Info(fmt.Sprintf("Unknown owner kind: %v", owner.Kind))
		}
	}

	return -1, errors.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
}

// GetPodRank returns the rank of this pod
func (vs *VineyardScheduling) GetPodRank(pod *v1.Pod, replica int) int {
	// get workflow's prefix name.
	podName := pod.GetName()
	prefixIndex := strings.LastIndexByte(podName, '-')
	prefixName := podName[:prefixIndex]

	// clean up the pod rank
	if len(vs.podRank[prefixName]) > replica {
		delete(vs.podRank, prefixName)
	}

	rank, prefixExist := vs.podRank[prefixName]
	if !prefixExist {
		m := make(map[string]int)
		m[podName] = len(rank)
		vs.podRank[prefixName] = m
		return 0
	} else {
		_, podExist := rank[podName]
		if !podExist {
			rank[podName] = len(rank)
		}
	}

	return rank[podName]
}

// GetNamespacedName returns the namespaced name of an kubernetes object.
func GetNamespacedName(object metav1.Object) string {
	return fmt.Sprintf("%v/%v", object.GetNamespace(), object.GetName())
}
