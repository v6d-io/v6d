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
	listerv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
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
	podLister       listerv1.PodLister
	scheduleTimeout *time.Duration
	state           map[string]*SchedulerState
	podRank         map[string]map[string]int64
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
	state := make(map[string]*SchedulerState)
	scheduling := &VineyardScheduling{
		Client:          client,
		handle:          handle,
		podLister:       handle.SharedInformerFactory().Core().V1().Pods().Lister(),
		scheduleTimeout: &timeout,
		state:           state,
		podRank:         map[string]map[string]int64{},
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
	job, replica, requires, vineyardd, err := vs.GetJobInfo(pod)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	slog.Info(
		fmt.Sprintf(
			"scoring for pod of job %v, with %v replicas, and requires %v",
			job,
			replica,
			requires,
		),
	)

	schedulerState := vs.MakeSchedulerStateForNamespace(VineyardSystemNamespace)
	podRank := vs.GetPodRank(pod, replica)
	nodes := vs.GetAllWorkerNodes(vineyardd)

	score, err := schedulerState.Compute(ctx, job, replica, podRank, nodes, requires, nodeName, pod)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}
	if score == 0 {
		return score, framework.NewStatus(framework.Unschedulable, "Computed store is zero")
	}
	slog.Info(fmt.Sprintf("score for pod of job %v on node %v is: %v", job, nodeName, score))
	return score, framework.NewStatus(framework.Success, "")
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

// MakeSchedulerStateForNamespace initializes a state for the given namespace, if not exists.
func (vs *VineyardScheduling) MakeSchedulerStateForNamespace(namespace string) *SchedulerState {
	if _, ok := vs.state[namespace]; !ok {
		state := make(map[string]map[string]string)
		vs.state[namespace] = &SchedulerState{
			Client: vs.Client,
			state:  state,
		}
	}
	return vs.state[namespace]
}

func (vs *VineyardScheduling) getJobName(pod *v1.Pod) (string, error) {
	jobName, exists := pod.Labels[labels.VineyardJobName]
	slog.Info(fmt.Sprintf("labels: %v", pod.Labels))
	if !exists || jobName == "" {
		return "", errors.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
	}
	slog.Info(fmt.Sprintf("Get job's name: %v", jobName))
	return jobName, nil
}

func (vs *VineyardScheduling) getJobReplica(pod *v1.Pod) (int64, error) {
	// infer from the ownership
	ctx := context.TODO()
	// ctx := context.Background()
	for _, owner := range pod.GetOwnerReferences() {
		name := types.NamespacedName{Namespace: pod.Namespace, Name: owner.Name}
		switch owner.Kind {
		case "ReplicaSet":
			replicaset := &appsv1.ReplicaSet{}
			if err := vs.Get(ctx, name, replicaset); err == nil {
				return int64(*replicaset.Spec.Replicas), nil
			}
		case "DaemonSet":
			daemonset := &appsv1.DaemonSet{}
			if err := vs.Get(ctx, name, daemonset); err == nil {
				return int64(daemonset.Spec.Size()), nil
			}
		case "StatefulSet":
			statefulset := &appsv1.StatefulSet{}
			if err := vs.Get(ctx, name, statefulset); err == nil {
				return int64(*statefulset.Spec.Replicas), nil
			}
		case "Job":
			job := &batchv1.Job{}
			if err := vs.Get(ctx, name, job); err == nil {
				return int64(*job.Spec.Parallelism), nil
			}
		case "CronJob":
			crobjob := &batchv1.CronJob{}
			if err := vs.Get(ctx, name, crobjob); err == nil {
				return int64(crobjob.Spec.Size()), nil
			}
		case "Deployment":
			deployment := &appsv1.Deployment{}
			if err := vs.Get(ctx, name, deployment); err == nil {
				return int64(*deployment.Spec.Replicas), nil
			}
		default:
			slog.Info(fmt.Sprintf("Unknown owner kind: %v", owner.Kind))
		}
	}

	return -1, errors.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
}

// GetAllWorkerNodes records every worker node which deployed vineyardd.
func (vs *VineyardScheduling) GetAllWorkerNodes(vineyardd string) []string {
	name := ParseNamespacedName(vineyardd)
	return GetVineyarddNodes(vs.Client, slog, name.Name, name.Namespace)
}

// GetJobInfo requires (job, replica, requires, vineyardd) information of a pod.
func (vs *VineyardScheduling) GetJobInfo(pod *v1.Pod) (string, int64, []string, string, error) {
	job, err := vs.getJobName(pod)
	if err != nil {
		return "", 0, nil, "", err
	}
	replica, err := vs.getJobReplica(pod)
	if err != nil {
		return "", 0, nil, "", err
	}
	requires, err := GetRequiredJob(slog, pod.Annotations)
	if err != nil {
		return "", 0, nil, "", err
	}
	vineyardd, exist := pod.Labels[labels.VineyarddName]
	if !exist {
		slog.Info("VineyarddName does't exist!")
	}
	vineyarddNS := pod.Labels[labels.VineyarddNamespace]
	var name string
	if vineyarddNS == "" {
		name = vineyardd
	} else {
		name = vineyarddNS + "/" + vineyardd
	}
	return job, replica, requires, name, nil
}

// GetPodRank returns the rank of this pod
func (vs *VineyardScheduling) GetPodRank(pod *v1.Pod, replica int64) int64 {
	// get workflow's prefix name.
	podName := pod.GetName()
	prefixIndex := strings.LastIndexByte(podName, '-')
	prefixName := podName[:prefixIndex]

	// clean up the pod rank
	if int64(len(vs.podRank[prefixName])) > replica {
		delete(vs.podRank, prefixName)
	}

	rank, prefixExist := vs.podRank[prefixName]
	if !prefixExist {
		m := make(map[string]int64)
		m[podName] = int64(len(rank))
		vs.podRank[prefixName] = m
		return 0
	} else {
		_, podExist := rank[podName]
		if !podExist {
			rank[podName] = int64(len(rank))
		}
	}

	return rank[podName]
}

// GetNamespacedName returns the namespaced name of an kubernetes object.
func GetNamespacedName(object metav1.Object) string {
	return fmt.Sprintf("%v/%v", object.GetNamespace(), object.GetName())
}

func ParseNamespacedName(name string, defaultNamespace ...string) types.NamespacedName {
	separator := string(types.Separator)
	if strings.Contains(name, separator) {
		split := strings.SplitN(name, separator, 2)
		return types.NamespacedName{
			Namespace: split[0],
			Name:      split[1],
		}
	} else {
		if len(defaultNamespace) > 0 {
			return types.NamespacedName{
				Namespace: defaultNamespace[0],
				Name:      name,
			}
		} else {
			return types.NamespacedName{
				Name: name,
			}
		}
	}
}
