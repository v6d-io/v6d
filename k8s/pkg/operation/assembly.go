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
	"context"
	"strings"

	"github.com/pkg/errors"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
)

const (
	// PodNameLabelKey is the label key for the pod name which generated the stream
	PodNameLabelKey = "k8s.v6d.io/created-podname"
	// PodNameSpaceLabelKey is the label key for the pod namespace which generated the stream
	PodNameSpaceLabelKey = "k8s.v6d.io/created-podnamespace"
	// NeedInjectedAssemblyKey represents the pod need to be injected with the assembly job
	NeedInjectedAssemblyKey = "need-injected-assembly"
	// AssemblyPrefix is the prefix of the assembly job
	AssemblyPrefix = "assemble-"
)

// AssemblyOperation is the operation for the assembly
type AssemblyOperation struct {
	client.Client
	ClientUtils
	app  *kubernetes.Application
	done bool
}

// LocalAssemblyConfig is the config for the local assembly job
type LocalAssemblyConfig struct {
	Name               string
	Namespace          string
	StreamID           string
	JobName            string
	NodeName           string
	TimeoutSeconds     int64
	VineyardSockPath   string
	LocalAssemblyImage string
}

// DistributedAssemblyConfig is the config for the distributed assembly job
type DistributedAssemblyConfig struct {
	Name                     string
	Namespace                string
	StreamID                 string
	JobName                  string
	GlobalObjectID           string
	OldObjectToNewObject     string
	TimeoutSeconds           int64
	VineyardSockPath         string
	DistributedAssemblyImage string
}

// LocalAssemblyConfigTemplate is the template config for the assembly job
var LocalAssemblyConfigTemplate LocalAssemblyConfig

// GetAssemblyConfig gets the local assembly config
func GetAssemblyConfig() LocalAssemblyConfig {
	return LocalAssemblyConfigTemplate
}

// DistributedAssemblyConfigTemplate is the template config for the distributed assembly job
var DistributedAssemblyConfigTemplate DistributedAssemblyConfig

// GetDistributedAssemblyConfig gets the distributed assembly config
func GetDistributedAssemblyConfig() DistributedAssemblyConfig {
	return DistributedAssemblyConfigTemplate
}

// CreateJob creates the job for the operation
func (ao *AssemblyOperation) CreateJob(ctx context.Context, o *v1alpha1.Operation) error {
	if err := ao.applyLocalAssemblyJob(ctx, o); err != nil {
		return errors.Wrap(err, "failed to apply the local assembly job")
	}

	done, err := ao.checkLocalAssemblyJob(ctx, o)
	if err != nil {
		return errors.Wrap(err, "failed to check the local assembly job")
	}

	if done {
		if o.Spec.Type == "local" {
			ao.done = true
			return nil
		}

		// all local assembly job done, then do distributed assembly job next
		if err := ao.applyDistributedAssemblyJob(ctx, o.Spec.TimeoutSeconds); err != nil {
			return errors.Wrap(err, "failed to apply the distributed assembly job")
		}

		distributedDone, err := ao.checkDistributedAssemblyJob(ctx, o)
		if err != nil {
			return errors.Wrap(err, "failed to check the distributed assembly job")
		}

		// local assembly job and distributed assembly job all done, then the operation is done
		if distributedDone {
			ao.done = true
		}
	}
	return nil
}

// findNeedAssemblyPodByLocalObject finds the pod which need to be injected with the local assembly job
func (ao *AssemblyOperation) findNeedAssemblyPodByLocalObject(
	ctx context.Context,
	labels *map[string]string,
) (*corev1.Pod, error) {
	podName := (*labels)[PodNameLabelKey]
	podNamespace := (*labels)[PodNameSpaceLabelKey]
	if podName != "" && podNamespace != "" {
		pod := &corev1.Pod{}
		if err := ao.Client.Get(ctx, client.ObjectKey{
			Name:      podName,
			Namespace: podNamespace,
		}, pod); err != nil {
			return nil, errors.Wrap(err, "failed to get the pod")
		}
		if v, ok := pod.Labels[NeedInjectedAssemblyKey]; ok && strings.ToLower(v) == "true" {
			return pod, nil
		}
	}
	return nil, nil
}

// buildLocalAssemblyJob build all configuration for the local assembly job
func (ao *AssemblyOperation) buildLocalAssemblyJob(
	ctx context.Context,
	localObject *v1alpha1.LocalObject,
	pod *corev1.Pod,
	timeout int64,
) error {
	podLabels := pod.Labels

	vineyarddName := podLabels[labels.VineyarddName]
	vineyarddNamespace := podLabels[labels.VineyarddNamespace]
	// get vineyardd cluster info
	vineyardd := &v1alpha1.Vineyardd{}
	if err := ao.Client.Get(ctx, client.ObjectKey{
		Name:      vineyarddName,
		Namespace: vineyarddNamespace,
	}, vineyardd); err != nil {
		return errors.Wrap(err, "failed to get the vineyardd")
	}

	// When the pod which generated the stream is annotated, the assembly job will be created in the same pod
	if _, ok := podLabels[NeedInjectedAssemblyKey]; ok {
		if strings.Contains(strings.ToLower(localObject.Spec.Typename), "stream") {
			LocalAssemblyConfigTemplate.Name = AssemblyPrefix + localObject.Spec.ObjectID
			LocalAssemblyConfigTemplate.Namespace = pod.Namespace
			LocalAssemblyConfigTemplate.StreamID = localObject.Spec.ObjectID
			LocalAssemblyConfigTemplate.NodeName = localObject.Spec.Hostname
			LocalAssemblyConfigTemplate.JobName = podLabels[labels.VineyardJobName]
			LocalAssemblyConfigTemplate.TimeoutSeconds = timeout
			LocalAssemblyConfigTemplate.LocalAssemblyImage = vineyardd.Spec.PluginImage.LocalAssemblyImage
			if socket, err := ao.ResolveRequiredVineyarddSocket(
				ctx,
				podLabels[labels.VineyarddName],
				podLabels[labels.VineyarddNamespace],
				localObject.Namespace,
			); err != nil {
				return err
			} else {
				LocalAssemblyConfigTemplate.VineyardSockPath = socket
			}
		}
	}
	return nil
}

// applyLocalAssemblyJob will apply the local assembly job
func (ao *AssemblyOperation) applyLocalAssemblyJob(
	ctx context.Context,
	o *v1alpha1.Operation,
) error {
	localObjectList := &v1alpha1.LocalObjectList{}

	if err := ao.Client.List(ctx, localObjectList); err != nil {
		return errors.Wrap(err, "failed to list the local objects")
	}

	for i := range localObjectList.Items {
		pod, err := ao.findNeedAssemblyPodByLocalObject(ctx, &localObjectList.Items[i].Labels)
		if err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return errors.Wrap(
				err,
				"failed to find the pod which needs to be injected with the assembly job",
			)
		}
		if pod != nil {
			if err := ao.buildLocalAssemblyJob(ctx, &localObjectList.Items[i], pod, o.Spec.TimeoutSeconds); err != nil {
				return errors.Wrap(err, "failed to build the local assembly job")
			}
			if _, err := ao.app.Apply(ctx, "operation/local-assembly-job.yaml", ctrl.Log, false); err != nil {
				return errors.Wrap(err, "failed to apply the local assembly job")
			}
		}
	}

	return nil
}

// checkLocalAssemblyJob will check the local assembly job's status
func (ao *AssemblyOperation) checkLocalAssemblyJob(
	ctx context.Context,
	o *v1alpha1.Operation,
) (bool, error) {
	podList := &corev1.PodList{}
	opts := []client.ListOption{
		client.MatchingLabels{
			labels.AssemblyEnabledLabel: "true",
		},
	}

	if err := ao.Client.List(ctx, podList, opts...); err != nil {
		return false, errors.Wrap(err, "failed to list the pods")
	}

	localObjectList := &v1alpha1.LocalObjectList{}

	// get all localobjects which may need to be injected with the assembly job
	if err := ao.Client.List(ctx, localObjectList); err != nil {
		return false, errors.Wrap(err, "failed to list the local objects")
	}

	targetLocalObjects := map[string]bool{}
	for i := range localObjectList.Items {
		job := &batchv1.Job{}
		if strings.Contains(strings.ToLower(localObjectList.Items[i].Spec.Typename), "stream") {
			if err := ao.Client.Get(ctx, client.ObjectKey{
				Name:      AssemblyPrefix + localObjectList.Items[i].Spec.ObjectID,
				Namespace: o.Namespace,
			}, job); err != nil {
				return false, errors.Wrap(err, "failed to get the job")
			}
			targetLocalObjects[localObjectList.Items[i].Spec.ObjectID] = true
			// if the job failed, then return false
			if job.Status.Succeeded == 0 {
				return false, nil
			}
		}
	}

	if err := ao.UpdateConfigmap(ctx, targetLocalObjects, o, AssemblyPrefix, &map[string]string{}); err != nil {
		return false, errors.Wrap(err, "failed to update the configmap")
	}

	return true, nil
}

// buildDistributedAssemblyJob build all configuration for the distributed assembly job
func (ao *AssemblyOperation) buildDistributedAssemblyJob(
	ctx context.Context,
	globalObject *v1alpha1.GlobalObject,
	pod *corev1.Pod,
	timeout int64,
) (bool, error) {
	podLabels := pod.Labels
	signatures := map[string]bool{}
	for i := range globalObject.Spec.Members {
		signatures[globalObject.Spec.Members[i]] = true
	}

	localObjectList := &v1alpha1.LocalObjectList{}
	if err := ao.Client.List(ctx, localObjectList); err != nil {
		return false, errors.Wrap(err, "failed to list the local objects")
	}

	sigToID := map[string]string{}
	for i := range localObjectList.Items {
		if _, ok := signatures[localObjectList.Items[i].Spec.Signature]; ok &&
			strings.Contains(strings.ToLower(localObjectList.Items[i].Spec.Typename), "stream") {
			sigToID[localObjectList.Items[i].Spec.Signature] = localObjectList.Items[i].Name
		}
	}

	globalObjectList := &v1alpha1.GlobalObjectList{}
	if err := ao.Client.List(ctx, globalObjectList); err != nil {
		return false, errors.Wrap(err, "failed to list the global objects")
	}

	oldObjectToNewObject := map[string]string{}
	for i := range globalObjectList.Items {
		labels := globalObjectList.Items[i].Labels
		if v, ok := labels[PodNameLabelKey]; ok {
			for j := range sigToID {
				if strings.Contains(v, sigToID[j]) {
					oldObjectToNewObject[sigToID[j]] = globalObjectList.Items[i].Name
				}
			}
		}
	}

	vineyarddName := podLabels[labels.VineyarddName]
	vineyarddNamespace := podLabels[labels.VineyarddNamespace]
	// get vineyardd cluster info
	vineyardd := &v1alpha1.Vineyardd{}
	if err := ao.Client.Get(ctx, client.ObjectKey{
		Name:      vineyarddName,
		Namespace: vineyarddNamespace,
	}, vineyardd); err != nil {
		return false, errors.Wrap(err, "failed to get the vineyardd")
	}

	// build the distributed assembly job
	if len(sigToID) == len(oldObjectToNewObject) {
		str := `'{`
		for k, v := range oldObjectToNewObject {
			str = str + `"` + k + `"` + ":" + `"` + v + `"` + ","
		}
		str = str[:len(str)-1] + `}'`
		DistributedAssemblyConfigTemplate.Name = AssemblyPrefix + globalObject.Name
		DistributedAssemblyConfigTemplate.Namespace = pod.Namespace
		DistributedAssemblyConfigTemplate.GlobalObjectID = globalObject.Name
		DistributedAssemblyConfigTemplate.OldObjectToNewObject = str
		DistributedAssemblyConfigTemplate.JobName = podLabels[labels.VineyardJobName]
		DistributedAssemblyConfigTemplate.TimeoutSeconds = timeout
		DistributedAssemblyConfigTemplate.DistributedAssemblyImage = vineyardd.Spec.PluginImage.DistributedAssemblyImage
		if socket, err := ao.ResolveRequiredVineyarddSocket(
			ctx,
			podLabels[labels.VineyarddName],
			podLabels[labels.VineyarddNamespace],
			globalObject.Namespace,
		); err != nil {
			return false, nil
		} else {
			DistributedAssemblyConfigTemplate.VineyardSockPath = socket
		}
		return true, nil
	}
	return false, nil
}

// applyDistributedAssemblyJob will apply the distributed assembly job
func (ao *AssemblyOperation) applyDistributedAssemblyJob(ctx context.Context, timeout int64) error {
	globalObjectList := &v1alpha1.GlobalObjectList{}
	if err := ao.Client.List(ctx, globalObjectList); err != nil {
		return errors.Wrap(err, "failed to list the local objects")
	}

	for i := range globalObjectList.Items {
		pod, err := ao.findNeedAssemblyPodByLocalObject(ctx, &globalObjectList.Items[i].Labels)
		if err != nil {
			return errors.Wrap(
				err,
				"failed to find the pod which needs to be injected with the assembly job",
			)
		}
		if pod != nil {
			needJob, err := ao.buildDistributedAssemblyJob(
				ctx,
				&globalObjectList.Items[i],
				pod,
				timeout,
			)
			if err != nil {
				return errors.Wrap(err, "failed to build the distributed assembly job")
			}
			if needJob {
				if _, err := ao.app.Apply(ctx, "operation/distributed-assembly-job.yaml", ctrl.Log, false); err != nil {
					return errors.Wrap(err, "failed to apply the global assembly job")
				}
			}
		}
	}

	return nil
}

// checkDistributedAssemblyJob will check the status of distributed assembly job and local assembly.
func (ao *AssemblyOperation) checkDistributedAssemblyJob(
	ctx context.Context,
	o *v1alpha1.Operation,
) (bool, error) {
	required := o.Spec.Require
	jobNames := strings.Split(required, ".")
	jobNameToPodName := map[[2]string]bool{}
	for i := range jobNames {
		podList := &corev1.PodList{}
		opts := []client.ListOption{
			client.MatchingLabels{
				"app": jobNames[i],
			},
		}

		if err := ao.Client.List(ctx, podList, opts...); err != nil {
			return false, errors.Wrap(err, "failed to list the pods")
		}

		for _, p := range podList.Items {
			jobNameToPodName[[2]string{p.Name, p.Namespace}] = true
		}
	}

	globalObjectList := &v1alpha1.GlobalObjectList{}

	// get all globalobjects which may need to be injected with the assembly job
	if err := ao.Client.List(ctx, globalObjectList); err != nil {
		return false, errors.Wrap(err, "failed to list the global objects")
	}

	if len(globalObjectList.Items) == 0 {
		return false, nil
	}
	targetGlobalObjects := map[string]bool{}
	for i := range globalObjectList.Items {
		labels := globalObjectList.Items[i].Labels
		job := &batchv1.Job{}
		if jobNameToPodName[[2]string{labels[PodNameLabelKey], labels[PodNameSpaceLabelKey]}] &&
			strings.Contains(strings.ToLower(globalObjectList.Items[i].Spec.Typename), "stream") {
			if err := ao.Client.Get(ctx, client.ObjectKey{Name: AssemblyPrefix + globalObjectList.Items[i].Spec.ObjectID, Namespace: o.Namespace}, job); err != nil {
				return false, errors.Wrap(err, "failed to get the job")
			}

			targetGlobalObjects[globalObjectList.Items[i].Spec.ObjectID] = true
			// if the job failed, then the distributed assembly job is failed
			if job.Status.Succeeded == 0 {
				return false, nil
			}
		}
	}

	if err := ao.UpdateConfigmap(ctx, targetGlobalObjects, o, AssemblyPrefix, &map[string]string{}); err != nil {
		return false, errors.Wrap(err, "failed to update the configmap")
	}

	return true, nil
}

// IsDone will check if the assembly operation is done
func (ao *AssemblyOperation) IsDone() bool {
	return ao.done
}
