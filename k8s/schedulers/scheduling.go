/*
Copyright 2020 The Kubernetes Authors.
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

package schedulers

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"

	listerv1 "k8s.io/client-go/listers/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"

	v1alpha1 "github.com/v6d-io/v6d/k8s/api/k8s/v1alpha1"
	clientset "github.com/v6d-io/v6d/k8s/generated/clientset/versioned"
	clientsetv1alpha1 "github.com/v6d-io/v6d/k8s/generated/clientset/versioned/typed/k8s/v1alpha1"

	_ "github.com/v6d-io/v6d/k8s/generated/informers/externalversions/k8s/v1alpha1"
	_ "github.com/v6d-io/v6d/k8s/generated/listers/k8s/v1alpha1"
)

// SchedulerState records the status of current scheduling
type SchedulerState struct {
	state     map[string]map[string]string // { jobname: { pod: nodename }}
	localctl  clientsetv1alpha1.LocalObjectInterface
	globalctl clientsetv1alpha1.GlobalObjectInterface
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

// Compute the placement of a pod in job, assuming the useable nodes, and based on the given objects pool.
//
// Use a deterministic strategy.
func (ss *SchedulerState) Compute(ctx context.Context, job string, replica int64, rank int64, requires []string, nodeName string) (int64, error) {
	// if requires no vineyard object, raise
	if len(requires) == 0 {
		return 0, fmt.Errorf("No nodes available")
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
		for _, sig := range globalObject.Spec.Members {
			localsigs = append(localsigs, sig)
		}
	}
	localObjects, err := ss.getLocalObjectsBySignatures(ctx, localsigs)
	if err != nil {
		return 0, err
	}
	if len(localObjects) == 0 {
		return 0, fmt.Errorf("No local chunks found")
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

	var cnt int64 = 0
	target := ""
	for _, node := range nodes {
		localfrags := int64(len(locations[node]))
		if cnt+localfrags >= (nchunks*rank + (nchunks+1)/2) {
			target = node
			break
		}
		cnt += localfrags
	}

	if target == "" {
		klog.V(5).Infof("Unable to find a target: replica = %v, rank = %v, locations = %v", replica, rank, locations)
		return 0, fmt.Errorf("Unable to find a pod: internal error")
	}

	if target == nodeName {
		return 100, nil
	} else {
		return 1, nil
	}
}

func (ss *SchedulerState) getGlobalObjectsByID(ctx context.Context, objectIds []string) ([]*v1alpha1.GlobalObject, error) {
	objects := make([]*v1alpha1.GlobalObject, 0)
	for _, globalObjectID := range objectIds {
		if globalObject, err := ss.globalctl.Get(ctx, globalObjectID, metav1.GetOptions{}); err != nil {
			return nil, err
		} else {
			objects = append(objects, globalObject)
		}
	}
	return objects, nil
}

func (ss *SchedulerState) getLocalObjectsBySignatures(ctx context.Context, signatures []string) ([]*v1alpha1.LocalObject, error) {
	objects := make([]*v1alpha1.LocalObject, 0)
	for _, sig := range signatures {
		options := metav1.ListOptions{
			LabelSelector: fmt.Sprintf("k8s.v6d.io/signature=%v", sig),
		}
		if localObjects, err := ss.localctl.List(ctx, options); err != nil {
			return nil, err
		} else {
			for _, localObject := range localObjects.Items {
				objects = append(objects, &localObject)
			}
		}
	}
	return objects, nil
}

// VineyardScheduling is a plugin that schedules pods that requires vineyard objects as inputs.
type VineyardScheduling struct {
	handle          framework.FrameworkHandle
	podLister       listerv1.PodLister
	scheduleTimeout *time.Duration
	state           map[string]*SchedulerState
	client          *clientset.Clientset
}

var _ framework.ScorePlugin = &VineyardScheduling{}
var _ framework.PreFilterPlugin = &VineyardScheduling{}

// var _ framework.PermitPlugin = &VineyardScheduling{}
var _ framework.PostBindPlugin = &VineyardScheduling{}

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
)

// New initializes a vineyard scheduler
// func New(obj runtime.Object, handle framework.FrameworkHandle) (framework.Plugin, error) {
func New(configuration runtime.Object, handle framework.FrameworkHandle) (framework.Plugin, error) {
	klog.Info("Initializing the vineyard scheduler plugin ...")
	timeout := Timeout * time.Second
	state := make(map[string]*SchedulerState)
	client := clientset.NewForConfigOrDie(ctrl.GetConfigOrDie())
	scheduling := &VineyardScheduling{
		handle:          handle,
		podLister:       handle.SharedInformerFactory().Core().V1().Pods().Lister(),
		scheduleTimeout: &timeout,
		state:           state,
		client:          client,
	}
	return scheduling, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (vs *VineyardScheduling) Name() string {
	return Name
}

// Less compares the priority of two
// func (vs *VineyardScheduling) Less(pod1, pod2 *framework.QueuedPodInfo) bool {
func (vs *VineyardScheduling) Less(pod1, pod2 *framework.PodInfo) bool {
	prio1 := podutil.GetPodPriority(pod1.Pod)
	prio2 := podutil.GetPodPriority(pod2.Pod)
	return prio1 > prio2
}

// PreFilter for a pod
func (vs *VineyardScheduling) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) *framework.Status {
	return framework.NewStatus(framework.Success, "")
}

// PreFilterExtensions is None
func (vs *VineyardScheduling) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// Score compute the score for a pod based on the status of required vineyard objects.
//
func (vs *VineyardScheduling) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	// nodeInfo, err := ps.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	// if err != nil {
	// 	return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Faild to get node %q: %v", nodeName, err))
	// }

	klog.V(5).Infof("scoring for pod %v on node %v", GetNamespacedName(pod), nodeName)

	job, replica, requires, err := vs.GetVineyardLabels(pod)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	rank, err := vs.GetPodRank(pod)
	if err != nil || rank == -1 {
		rank = replica - 1
	}

	klog.V(5).Infof("scoring for pod of job %v, with %v replicas (rank %v), and requires %v", job, replica, rank, requires)

	namespace := pod.GetNamespace()
	schedulerState := vs.MakeSchedulerStateForNamespace(namespace)

	score, err := schedulerState.Compute(ctx, job, replica, rank, requires, nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Unschedulable, err.Error())
	}
	klog.Infof("score for pod of job %v on node %v is: %v", job, nodeName, score)
	return score, framework.NewStatus(framework.Success, "")
}

// ScoreExtensions of the Score plugin.
func (vs *VineyardScheduling) ScoreExtensions() framework.ScoreExtensions {
	return vs
}

// NormalizeScore normalizes the score of all nodes for a pod.
func (vs *VineyardScheduling) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	// Find highest and lowest scores.

	return framework.NewStatus(framework.Success, "")
}

// Permit only permit runs on the node that has vineyard installed.
func (vs *VineyardScheduling) Permit(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (*framework.Status, time.Duration) {
	return framework.NewStatus(framework.Success, ""), 0
}

// PostBind do nothing
func (vs *VineyardScheduling) PostBind(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeName string) {
	klog.V(5).Infof("bind pod %v on node %v", GetNamespacedName(pod), nodeName)
	job, replica, requires, err := vs.GetVineyardLabels(pod)
	if err != nil {
		// ignore: might not be a vineyard job
		return
	}
	klog.V(5).Infof("bind pod of job %v, with %v replicas, and requires %v", job, replica, requires)

	// ignore
	//
	// namespace := pod.GetNamespace()
	// schedulerState := vs.MakeSchedulerStateForNamespace(namespace)
	// schedulerState.Append(job, GetNamespacedName(pod), nodeName)
}

// MakeSchedulerStateForNamespace initializes a state for the given namespace, if not exists.
func (vs *VineyardScheduling) MakeSchedulerStateForNamespace(namespace string) *SchedulerState {
	if _, ok := vs.state[namespace]; !ok {
		state := make(map[string]map[string]string)
		localctl := vs.client.K8sV1alpha1().LocalObjects(namespace)
		globalctl := vs.client.K8sV1alpha1().GlobalObjects(namespace)
		vs.state[namespace] = &SchedulerState{
			state:     state,
			localctl:  localctl,
			globalctl: globalctl,
		}
	}
	return vs.state[namespace]
}

func (vs *VineyardScheduling) getJobName(pod *v1.Pod) (string, error) {
	jobName, exist := pod.Labels[VineyardJobName]
	if !exist || jobName == "" {
		return "", fmt.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
	}
	klog.V(5).Infof("job name is: %v", jobName)
	return jobName, nil
}

func (vs *VineyardScheduling) getJobReplica(pod *v1.Pod) (int64, error) {
	jobReplica, exist := pod.Labels[VineyardJobReplica]
	if !exist || jobReplica == "" {
		return -1, fmt.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
	}
	klog.V(5).Infof("job replica is: %v", jobReplica)
	if val, err := strconv.Atoi(jobReplica); err != nil {
		return -1, err
	} else {
		return int64(val), nil
	}
}

func (vs *VineyardScheduling) getJobRequired(pod *v1.Pod) ([]string, error) {
	objects, exist := pod.Labels[VineyardJobRequired]
	if !exist || objects == "" {
		return nil, fmt.Errorf("Failed to get vineyard job name for %v", GetNamespacedName(pod))
	}
	klog.V(5).Infof("job requires: %v", objects)
	return strings.Split(objects, "-"), nil
}

// GetVineyardLabels requires (job, replica, requires) information of a pod.
func (vs *VineyardScheduling) GetVineyardLabels(pod *v1.Pod) (string, int64, []string, error) {
	job, err := vs.getJobName(pod)
	if err != nil {
		return "", 0, nil, err
	}
	replica, err := vs.getJobReplica(pod)
	if err != nil {
		return "", 0, nil, err
	}
	requires, err := vs.getJobRequired(pod)
	if err != nil {
		return "", 0, nil, err
	}
	return job, replica, requires, nil
}

// GetPodRank returns the rank of this pod
func (vs *VineyardScheduling) GetPodRank(pod *v1.Pod) (int64, error) {
	names := strings.Split(pod.GetName(), "-")
	if rank, err := strconv.Atoi(names[len(names)-1]); err != nil {
		return -1, err
	} else {
		return int64(rank), nil
	}
}

// GetNamespacedName returns the namespaced name of an kubernetes object.
func GetNamespacedName(object metav1.Object) string {
	return fmt.Sprintf("%v/%v", object.GetNamespace(), object.GetName())
}
