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

// Package operation contains the operation logic
package operation

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"
	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/schedulers"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// RepartitionPrefix is the prefix of the assembly job
	RepartitionPrefix = "repartition-"
	// NeedInjecteRepartitionKey represents the pod need to be injected with the repartition job
	NeedInjecteRepartitionKey = "need-injected-repartition"
)

// RepartitionOperation is the operation for the repartition
type RepartitionOperation struct {
	client.Client
	app  *swckkube.Application
	done bool
}

// DaskRepartitionConfig is the config for the dask repartition job
type DaskRepartitionConfig struct {
	Name             string
	Namespace        string
	GlobalobjectID   string
	Replicas         string
	InstanceToWorker string
	DaskScheduler    string
	JobName          string
	TimeoutSeconds   int64
	VineyardSockPath string
}

// TmpDaskRepartitionConfig is the template config for the dask repartition job
var TmpDaskRepartitionConfig DaskRepartitionConfig

// GetDaskRepartitionConfig gets the dask repartition config
func GetDaskRepartitionConfig() DaskRepartitionConfig {
	return TmpDaskRepartitionConfig
}

// CreateJob creates the job for the repartition
func (ro *RepartitionOperation) CreateJob(ctx context.Context, o *v1alpha1.Operation) error {
	switch o.Spec.Type {
	case "dask":
		if err := ro.applyDaskRepartitionJob(ctx, o); err != nil {
			return fmt.Errorf("failed to apply dask repartition job: %w", err)
		}
		daskRepartitionDone, err := ro.checkDaskRepartitionJob(ctx, o)
		if err != nil {
			return fmt.Errorf("failed to check dask repartition job: %w", err)
		}
		ro.done = daskRepartitionDone
	}
	return nil
}

// buildDaskRepartitionJob builds the dask repartition job
func (ro *RepartitionOperation) buildDaskRepartitionJob(ctx context.Context, globalObject *v1alpha1.GlobalObject,
	pod *corev1.Pod, o *v1alpha1.Operation) error {
	require := o.Spec.Require
	podList := &corev1.PodList{}
	podOpts := []client.ListOption{
		client.MatchingLabels{
			schedulers.VineyardJobName: require,
		},
	}

	// get all instance's hostname
	instanceToNode := make(map[int]string)
	if err := ro.Client.List(ctx, podList, podOpts...); err != nil {
		return fmt.Errorf("failed to list pods: %w", err)
	}

	localObjectList := &v1alpha1.LocalObjectList{}
	if err := ro.Client.List(ctx, localObjectList); err != nil {
		return fmt.Errorf("failed to list local objects: %w", err)
	}

	for _, lo := range localObjectList.Items {
		instanceToNode[lo.Spec.InstanceID] = lo.Spec.Hostname
	}
	// get dask workers's hostname and their real name
	daskHostnameToName := make(map[string]string)
	// convert the selector to map
	anno := pod.GetAnnotations()
	daskWorkerSelector := anno[schedulers.DaskWorkerSelector]
	allselectors := strings.Split(daskWorkerSelector, ",")
	selector := map[string]string{}
	for _, s := range allselectors {
		str := strings.Split(s, ":")
		selector[str[0]] = str[1]
	}

	daskWorkPodList := &corev1.PodList{}
	daskWorkPodOpts := []client.ListOption{
		client.MatchingLabels(selector),
	}
	if err := ro.Client.List(ctx, daskWorkPodList, daskWorkPodOpts...); err != nil {
		return fmt.Errorf("failed to list dask workers: %w", err)
	}

	for i := range daskWorkPodList.Items {
		workerName, err := ro.getWorkerNameFromPodLogs(&daskWorkPodList.Items[i])
		if err != nil {
			return fmt.Errorf("failed to get worker name from pod logs: %w", err)
		}
		daskHostnameToName[daskWorkPodList.Items[i].Spec.NodeName] = workerName
	}

	// build vineyard instance to dask worker name
	instanceToWorker := "'{"
	for instance, hostname := range instanceToNode {
		workerName, ok := daskHostnameToName[hostname]
		if ok {
			instanceToWorker = instanceToWorker + `"` + strconv.Itoa(instance) + `"` + ":" + `"` + workerName + `"` + ","
		}
	}
	instanceToWorker = instanceToWorker[:len(instanceToWorker)-1] + `}'`
	for _, dp := range daskWorkPodList.Items {
		daskHostnameToName[dp.Spec.NodeName] = dp.Status.PodIP
	}

	// get target replicas
	target := o.Spec.Target
	targetPodList := &corev1.PodList{}
	targetPodOpts := []client.ListOption{
		client.MatchingLabels{
			schedulers.VineyardJobName: target,
		},
	}

	if err := ro.Client.List(ctx, targetPodList, targetPodOpts...); err != nil {
		return fmt.Errorf("failed to list target pods: %w", err)
	}
	replicas, ok := targetPodList.Items[0].Labels[schedulers.WorkloadReplicas]
	if !ok {
		return fmt.Errorf("failed to get replicas from target jobs")
	}

	TmpDaskRepartitionConfig.Replicas = "'" + replicas + "'"
	TmpDaskRepartitionConfig.Name = RepartitionPrefix + globalObject.Name
	TmpDaskRepartitionConfig.Namespace = pod.Namespace
	TmpDaskRepartitionConfig.GlobalobjectID = globalObject.Name
	TmpDaskRepartitionConfig.DaskScheduler = "'" + anno[schedulers.DaskScheduler] + "'"
	TmpDaskRepartitionConfig.JobName = pod.Labels[schedulers.VineyardJobName]
	TmpDaskRepartitionConfig.InstanceToWorker = instanceToWorker
	TmpDaskRepartitionConfig.TimeoutSeconds = o.Spec.TimeoutSeconds

	vineyardd, ok := pod.Labels[schedulers.VineyarddName]
	if !ok {
		return fmt.Errorf("failed to get vineyardd name from pod labels")
	}
	TmpDaskRepartitionConfig.VineyardSockPath = "/var/run/vineyard-" + globalObject.Namespace + "-" + vineyardd
	return nil
}

// findNeedDaskRepartitionPodByGlobalObject finds the pod which needs the dask repartition from global objects
func (ro *RepartitionOperation) findNeedDaskRepartitionPodByGlobalObject(ctx context.Context, labels *map[string]string) (*corev1.Pod, error) {
	podName := (*labels)[PodNameLabelKey]
	podNamespace := (*labels)[PodNameSpaceLabelKey]
	if podName != "" && podNamespace != "" {
		pod := &corev1.Pod{}
		if err := ro.Client.Get(ctx, client.ObjectKey{Name: podName, Namespace: podNamespace}, pod); err != nil {
			return nil, fmt.Errorf("failed to get the pod: %v", err)
		}
		if v, ok := pod.Labels[NeedInjecteRepartitionKey]; ok && strings.ToLower(v) == "true" {
			return pod, nil
		}
	}
	return nil, nil

}

// applyDaskRepartitionJob applies the dask repartition job
func (ro *RepartitionOperation) applyDaskRepartitionJob(ctx context.Context, o *v1alpha1.Operation) error {
	globalObjectList := &v1alpha1.GlobalObjectList{}

	if err := ro.Client.List(ctx, globalObjectList); err != nil {
		return fmt.Errorf("failed to list global objects: %w", err)
	}

	for i := range globalObjectList.Items {
		pod, err := ro.findNeedDaskRepartitionPodByGlobalObject(ctx, &globalObjectList.Items[i].Labels)
		if err != nil {
			return fmt.Errorf("failed to find the pod which needs to be injected with the repartition job: %v", err)
		}
		if pod != nil {
			if err := ro.buildDaskRepartitionJob(ctx, &globalObjectList.Items[i], pod, o); err != nil {
				return fmt.Errorf("failed to build dask repartition job: %v", err)
			}
			if _, err := ro.app.Apply(ctx, "operation/dask-repartition.yaml", ctrl.Log, false); err != nil {
				return fmt.Errorf("failed to apply the dask repartition job: %v", err)
			}
		}
	}

	return nil
}

// checkDaskRepartitionJob checks whether the dask repartition job is ready
func (ro *RepartitionOperation) checkDaskRepartitionJob(ctx context.Context, o *v1alpha1.Operation) (bool, error) {
	// get all required pod
	require := o.Spec.Require
	allRequiredPods := map[string]bool{}
	podList := &corev1.PodList{}
	podOpts := []client.ListOption{
		client.MatchingLabels{
			schedulers.VineyardJobName: require,
		},
	}

	if err := ro.Client.List(ctx, podList, podOpts...); err != nil {
		return false, fmt.Errorf("failed to list pods: %w", err)
	}
	for i := range podList.Items {
		allRequiredPods[podList.Items[i].Name] = true
	}

	// get all globalobjects and check if the repartition job is done
	globalObjectList := &v1alpha1.GlobalObjectList{}
	if err := ro.Client.List(ctx, globalObjectList); err != nil {
		return false, fmt.Errorf("failed to list global objects: %w", err)
	}

	targetGlobalObjects := map[string]bool{}
	for i := range globalObjectList.Items {
		labels := globalObjectList.Items[i].Labels
		createdPod := labels[PodNameLabelKey]
		if allRequiredPods[createdPod] {
			job := &batchv1.Job{}
			if err := ro.Client.Get(ctx, client.ObjectKey{Name: RepartitionPrefix + globalObjectList.Items[i].Name, Namespace: o.Namespace}, job); err != nil {
				return false, fmt.Errorf("failed to get the repartition job: %v", err)
			}
			targetGlobalObjects[globalObjectList.Items[i].Spec.ObjectID] = true
			// if the job failed, then the dask repartition job is failed
			if job.Status.Succeeded == 0 {
				return false, nil
			}
		}
	}

	data := map[string]string{}
	data["InstanceToWorker"] = strings.Trim(TmpDaskRepartitionConfig.InstanceToWorker, "'")
	if err := UpdateConfigmap(ctx, ro.Client, targetGlobalObjects, o, RepartitionPrefix, &data); err != nil {
		return false, fmt.Errorf("failed to update the configmap: %v", err)
	}
	return true, nil
}

// getWorkerNameFromPodLogs get worker name from pod's logs
func (ro *RepartitionOperation) getWorkerNameFromPodLogs(pod *corev1.Pod) (string, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return "", fmt.Errorf("failed to get in cluster config: %v", err)
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return "", fmt.Errorf("failed to get clientset: %v", err)
	}
	req := clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{})
	logs, err := req.Stream(context.Background())
	if err != nil {
		return "", fmt.Errorf("failed to open stream: %v", err)
	}
	defer logs.Close()

	buf := new(bytes.Buffer)
	_, err = io.Copy(buf, logs)
	if err != nil {
		return "", fmt.Errorf("failed to copy logs: %v", err)
	}
	log := buf.String()

	// delete useless info
	workerIndex := strings.Index(log, "Start worker at")
	log = log[workerIndex:]

	ipPrefix := "tcp://" + pod.Status.PodIP + ":"
	start := strings.Index(log, ipPrefix)
	end := start
	for i := start + len(ipPrefix); i < len(log); i++ {
		if !unicode.IsDigit(rune(log[i])) {
			end = i
			break
		}
	}

	return log[start:end], nil
}

// IsDone checks whether the repartition operation is done
func (ro *RepartitionOperation) IsDone() bool {
	return ro.done
}
