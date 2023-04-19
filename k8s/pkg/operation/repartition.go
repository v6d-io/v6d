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

// Package operation contains the operation logic
package operation

import (
	"bytes"
	"context"
	"io"
	"strconv"
	"strings"
	"unicode"

	"github.com/pkg/errors"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
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
	*kubernetes.Clientset
	ClientUtils
	app  *swckkube.Application
	done bool
}

// DaskRepartitionConfig is the config for the dask repartition job
type DaskRepartitionConfig struct {
	Name                 string
	Namespace            string
	GlobalObjectID       string
	Replicas             string
	InstanceToWorker     string
	DaskScheduler        string
	JobName              string
	TimeoutSeconds       int64
	VineyardSockPath     string
	DaskRepartitionImage string
}

// DaskRepartitionConfigTemplate is the template config for the dask repartition job
var DaskRepartitionConfigTemplate DaskRepartitionConfig

// GetDaskRepartitionConfig gets the dask repartition config
func GetDaskRepartitionConfig() DaskRepartitionConfig {
	return DaskRepartitionConfigTemplate
}

// CreateJob creates the job for the repartition
func (ro *RepartitionOperation) CreateJob(ctx context.Context, o *v1alpha1.Operation) error {
	switch o.Spec.Type {
	case "dask":
		if err := ro.applyDaskRepartitionJob(ctx, o); err != nil {
			return errors.Wrap(err, "failed to apply dask repartition job")
		}
		daskRepartitionDone, err := ro.checkDaskRepartitionJob(ctx, o)
		if err != nil {
			return errors.Wrap(err, "failed to check dask repartition job")
		}
		ro.done = daskRepartitionDone
	}
	return nil
}

// buildDaskRepartitionJob builds the dask repartition job
func (ro *RepartitionOperation) buildDaskRepartitionJob(
	ctx context.Context,
	globalObject *v1alpha1.GlobalObject,
	pod *corev1.Pod,
	o *v1alpha1.Operation,
) error {
	require := o.Spec.Require
	podList := &corev1.PodList{}
	podOpts := []client.ListOption{
		client.MatchingLabels{
			labels.VineyardJobName: require,
		},
	}

	// get all instance's hostname
	instanceToNode := make(map[int]string)
	if err := ro.Client.List(ctx, podList, podOpts...); err != nil {
		return errors.Wrap(err, "failed to list pods")
	}

	localObjectList := &v1alpha1.LocalObjectList{}
	if err := ro.Client.List(ctx, localObjectList); err != nil {
		return errors.Wrap(err, "failed to list local objects")
	}

	for _, lo := range localObjectList.Items {
		instanceToNode[lo.Spec.InstanceID] = lo.Spec.Hostname
	}
	// get dask workers's hostname and their real name
	daskHostnameToName := make(map[string]string)
	// convert the selector to map
	anno := pod.GetAnnotations()
	daskWorkerSelector := anno[annotations.DaskWorkerSelector]
	allSelectors := strings.Split(daskWorkerSelector, ",")
	selector := map[string]string{}
	for _, s := range allSelectors {
		str := strings.Split(s, ":")
		selector[str[0]] = str[1]
	}

	daskWorkPodList := &corev1.PodList{}
	daskWorkPodOpts := []client.ListOption{
		client.MatchingLabels(selector),
	}
	if err := ro.Client.List(ctx, daskWorkPodList, daskWorkPodOpts...); err != nil {
		return errors.Wrap(err, "failed to list dask workers")
	}

	for i := range daskWorkPodList.Items {
		workerName, err := ro.getWorkerNameFromPodLogs(&daskWorkPodList.Items[i])
		if err != nil {
			return errors.Wrap(err, "failed to get worker name from pod logs")
		}
		daskHostnameToName[daskWorkPodList.Items[i].Spec.NodeName] = workerName
	}

	// build vineyard instance to dask worker name
	instanceToWorker := "'{"
	for instance, hostname := range instanceToNode {
		workerName, ok := daskHostnameToName[hostname]
		if ok {
			instanceToWorker = instanceToWorker + `"` + strconv.Itoa(
				instance,
			) + `"` + ":" + `"` + workerName + `"` + ","
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
			labels.VineyardJobName: target,
		},
	}

	if err := ro.Client.List(ctx, targetPodList, targetPodOpts...); err != nil {
		return errors.Wrap(err, "failed to list target pods")
	}
	replicas, ok := targetPodList.Items[0].Labels[labels.WorkloadReplicas]
	if !ok {
		return errors.New("failed to get replicas from target jobs")
	}

	vineyarddName := pod.Labels[labels.VineyarddName]
	vineyarddNamespace := pod.Labels[labels.VineyarddNamespace]
	// get vineyardd cluster info
	vineyardd := &v1alpha1.Vineyardd{}
	if err := ro.Client.Get(ctx, client.ObjectKey{
		Name:      vineyarddName,
		Namespace: vineyarddNamespace,
	}, vineyardd); err != nil {
		return errors.Wrap(err, "failed to get the vineyardd")
	}

	DaskRepartitionConfigTemplate.Replicas = "'" + replicas + "'"
	DaskRepartitionConfigTemplate.Name = RepartitionPrefix + globalObject.Name
	DaskRepartitionConfigTemplate.Namespace = pod.Namespace
	DaskRepartitionConfigTemplate.GlobalObjectID = globalObject.Name
	DaskRepartitionConfigTemplate.DaskScheduler = "'" + anno[annotations.DaskScheduler] + "'"
	DaskRepartitionConfigTemplate.JobName = pod.Labels[labels.VineyardJobName]
	DaskRepartitionConfigTemplate.InstanceToWorker = instanceToWorker
	DaskRepartitionConfigTemplate.TimeoutSeconds = o.Spec.TimeoutSeconds
	DaskRepartitionConfigTemplate.DaskRepartitionImage = vineyardd.Spec.PluginImage.DaskRepartitionImage
	if socket, err := ro.ResolveRequiredVineyarddSocket(
		ctx,
		pod.Labels[labels.VineyarddName],
		pod.Labels[labels.VineyarddNamespace],
		globalObject.Namespace,
	); err != nil {
		return nil
	} else {
		DaskRepartitionConfigTemplate.VineyardSockPath = socket
	}
	return nil
}

// findNeedDaskRepartitionPodByGlobalObject finds the pod which needs the dask repartition from global objects
func (ro *RepartitionOperation) findNeedDaskRepartitionPodByGlobalObject(
	ctx context.Context,
	labels *map[string]string,
) (*corev1.Pod, error) {
	podName := (*labels)[PodNameLabelKey]
	podNamespace := (*labels)[PodNameSpaceLabelKey]
	if podName != "" && podNamespace != "" {
		pod := &corev1.Pod{}
		if err := ro.Client.Get(ctx, client.ObjectKey{Name: podName, Namespace: podNamespace}, pod); err != nil {
			return nil, errors.Wrap(err, "failed to get the pod")
		}
		if v, ok := pod.Labels[NeedInjecteRepartitionKey]; ok && strings.ToLower(v) == "true" {
			return pod, nil
		}
	}
	return nil, nil
}

// applyDaskRepartitionJob applies the dask repartition job
func (ro *RepartitionOperation) applyDaskRepartitionJob(
	ctx context.Context,
	o *v1alpha1.Operation,
) error {
	globalObjectList := &v1alpha1.GlobalObjectList{}

	if err := ro.Client.List(ctx, globalObjectList); err != nil {
		return errors.Wrap(err, "failed to list global objects")
	}

	for i := range globalObjectList.Items {
		pod, err := ro.findNeedDaskRepartitionPodByGlobalObject(
			ctx,
			&globalObjectList.Items[i].Labels,
		)
		if err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return errors.Wrap(
				err,
				"failed to find the pod which needs to be injected with the repartition job",
			)
		}
		if pod != nil {
			if err := ro.buildDaskRepartitionJob(ctx, &globalObjectList.Items[i], pod, o); err != nil {
				return errors.Wrap(err, "failed to build dask repartition job")
			}
			if _, err := ro.app.Apply(ctx, "operation/dask-repartition.yaml", ctrl.Log, false); err != nil {
				return errors.Wrap(err, "failed to apply the dask repartition job")
			}
		}
	}

	return nil
}

// checkDaskRepartitionJob checks whether the dask repartition job is ready
func (ro *RepartitionOperation) checkDaskRepartitionJob(
	ctx context.Context,
	o *v1alpha1.Operation,
) (bool, error) {
	// get all required pod
	require := o.Spec.Require
	allRequiredPods := map[string]bool{}
	podList := &corev1.PodList{}
	podOpts := []client.ListOption{
		client.MatchingLabels{
			labels.VineyardJobName: require,
		},
	}

	if err := ro.Client.List(ctx, podList, podOpts...); err != nil {
		return false, errors.Wrap(err, "failed to list pods")
	}
	for i := range podList.Items {
		allRequiredPods[podList.Items[i].Name] = true
	}

	// get all globalobjects and check if the repartition job is done
	globalObjectList := &v1alpha1.GlobalObjectList{}
	if err := ro.Client.List(ctx, globalObjectList); err != nil {
		return false, errors.Wrap(err, "failed to list global objects")
	}

	targetGlobalObjects := map[string]bool{}
	for i := range globalObjectList.Items {
		labels := globalObjectList.Items[i].Labels
		createdPod := labels[PodNameLabelKey]
		if allRequiredPods[createdPod] {
			job := &batchv1.Job{}
			if err := ro.Client.Get(ctx, client.ObjectKey{Name: RepartitionPrefix + globalObjectList.Items[i].Name, Namespace: o.Namespace}, job); err != nil {
				return false, errors.Wrap(err, "failed to get the repartition job")
			}
			targetGlobalObjects[globalObjectList.Items[i].Spec.ObjectID] = true
			// if the job failed, then the dask repartition job is failed
			if job.Status.Succeeded == 0 {
				return false, nil
			}
		}
	}

	data := map[string]string{}
	data["InstanceToWorker"] = strings.Trim(DaskRepartitionConfigTemplate.InstanceToWorker, "'")
	if err := ro.UpdateConfigmap(ctx, targetGlobalObjects, o, RepartitionPrefix, &data); err != nil {
		return false, errors.Wrap(err, "failed to update the configmap")
	}
	return true, nil
}

// getWorkerNameFromPodLogs get worker name from pod's logs
func (ro *RepartitionOperation) getWorkerNameFromPodLogs(pod *corev1.Pod) (string, error) {
	req := ro.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{})
	logs, err := req.Stream(context.Background())
	if err != nil {
		return "", errors.Wrap(err, "failed to open stream")
	}
	defer logs.Close()

	buf := new(bytes.Buffer)
	_, err = io.Copy(buf, logs)
	if err != nil {
		return "", errors.Wrap(err, "failed to copy logs")
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
