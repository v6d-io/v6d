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
	"time"

	"k8s.io/klog/v2"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"k8s.io/component-helpers/scheduling/corev1"

	listerv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = "Vineyard"
	// Timeout is the default timeout for the scheduler plugin.
	Timeout = 60
	// VineyardJobName is the pod group name
	VineyardJobName = "scheduling.k8s.v6d.io/job"
	// VineyardJobRequired is the object ids that required by this job
	VineyardJobRequired = "scheduling.k8s.v6d.io/required"
	// VineyardJobReplica is the replication of pods in this job.
	VineyardJobReplica = "scheduling.k8s.v6d.io/replica"
	// ControlPlaneLabel is the label of the control plane
	ControlPlaneLabel = "node-role.kubernetes.io/control-plane"
	// VineyardSystemNamespace is the default system namespace
	VineyardSystemNamespace = "vineyard-system"
	// VineyarddName is the name of the vineyardd
	VineyarddName = "scheduling.k8s.v6d.io/vineyardd"
	// DaskScheduler is the name of the dask scheduler
	DaskScheduler = "scheduling.k8s.v6d.io/dask-scheduler"
	// DaskWorkerSelector is the selector of the dask worker
	DaskWorkerSelector = "scheduling.k8s.v6d.io/dask-worker-selector"
	// WorkloadReplicas is the replicas of workload
	WorkloadReplicas = "scheduling.k8s.v6d.io/replicas"
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
func New(client client.Client, config *rest.Config, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	klog.Info("Initializing the vineyard scheduler plugin ...")
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
func (vs *VineyardScheduling) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	klog.V(5).Infof("scoring for pod %v on node %v", GetNamespacedName(pod), nodeName)

	job, replica, requires, vineyardd, err := vs.GetJobInfo(pod)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	klog.V(5).Infof("scoring for pod of job %v, with %v replicas (rank %v), and requires %v", job, replica, requires)

	schedulerState := vs.MakeSchedulerStateForNamespace(VineyardSystemNamespace)
	podRank := vs.GetPodRank(pod, replica)
	nodes := vs.GetAllWorkerNodes(vineyardd)

	score, err := schedulerState.Compute(ctx, job, replica, podRank, nodes, requires, nodeName, pod)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}
	if score == 0 {
		return score, framework.NewStatus(framework.Unschedulable, "")
	}
	klog.Infof("score for pod of job %v on node %v is: %v", job, nodeName, score)
	return score, framework.NewStatus(framework.Success, "")
}

// ScoreExtensions of the Score plugin.
func (vs *VineyardScheduling) ScoreExtensions() framework.ScoreExtensions {
	klog.V(5).Infof("ScoreExtensions...")
	return vs
}

// NormalizeScore normalizes the score of all nodes for a pod.
func (vs *VineyardScheduling) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	klog.V(5).Infof("NormalizeScore...")
	// Find highest and lowest scores.
	return framework.NewStatus(framework.Success, "")
}

// Permit only permit runs on the node that has vineyard installed.
func (vs *VineyardScheduling) Permit(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (*framework.Status, time.Duration) {
	klog.V(5).Infof("Permit...")
	return framework.NewStatus(framework.Success, ""), 0
}

// PostBind prints the bind info
func (vs *VineyardScheduling) PostBind(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeName string) {
	klog.V(5).Infof("Bind pod %v on node %v", GetNamespacedName(pod), nodeName)
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
	jobName, exists := pod.Labels[VineyardJobName]
	klog.V(5).Infof("labels: %v", pod.Labels)
	if !exists || jobName == "" {
		return "", fmt.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
	}
	klog.V(5).Infof("Get job's name: %v", jobName)
	return jobName, nil
}

func (vs *VineyardScheduling) getJobReplica(pod *v1.Pod) (int64, error) {
	klog.V(5).Infof("getJobReplica...")
	// infer from the ownership
	ctx := context.TODO()
	//ctx := context.Background()
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
			klog.V(5).Infof("Unable to infer the job replica, unknown owner kind: %v", owner.Kind)
		}
	}

	return -1, fmt.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
}

// GetAllWorkerNodes records every worker node which deployed vineyardd.
func (vs *VineyardScheduling) GetAllWorkerNodes(vineyardd string) []string {
	nodes := []string{}

	podList := v1.PodList{}
	option := &client.ListOptions{
		LabelSelector: labels.SelectorFromSet(labels.Set{
			"app.kubernetes.io/name":     vineyardd,
			"app.kubernetes.io/instance": "vineyardd",
		}),
	}
	if err := vs.Client.List(context.TODO(), &podList, option); err != nil {
		klog.V(5).Infof("Failed to list all pods with the specific label: %v", err)
	}

	for _, pod := range podList.Items {
		nodes = append(nodes, pod.Spec.NodeName)
	}
	sort.Strings(nodes)
	return nodes
}

// get all required jobs name that separated by '.'
func (vs *VineyardScheduling) getRequiredJob(pod *v1.Pod) ([]string, error) {
	objects, exists := pod.Annotations[VineyardJobRequired]
	if !exists {
		return []string{}, fmt.Errorf("Failed to get the required jobs, please set none if there is no required job")
	}

	klog.V(5).Infof("Get the required jobs: %v", objects)
	if objects == "none" {
		return []string{}, nil
	}
	return strings.Split(objects, "."), nil
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
	requires, err := vs.getRequiredJob(pod)
	if err != nil {
		return "", 0, nil, "", err
	}
	vineyardd, exist := pod.Labels[VineyarddName]
	if !exist {
		klog.V(5).Infof("VineyarddName does't exist!")
	}
	return job, replica, requires, vineyardd, nil
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
