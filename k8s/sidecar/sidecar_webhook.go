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

// Package sidecar contains the logic for injecting vineyard sidecar
package sidecar

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// nolint: lll
// +kubebuilder:webhook:admissionReviewVersions=v1,sideEffects=None,path=/mutate-v1-pod-sidecar,mutating=true,failurePolicy=fail,groups="",resources=pods,verbs=create;update,versions=v1,name=mpod.sidecar.kb.io

// SidecarInjector injects vineyard sidecar container into Pods
type SidecarInjector struct {
	Client  client.Client
	decoder *admission.Decoder
}

const (
	// SidecarEnableLabel is the label key for enabling sidecar injection
	SidecarEnableLabel = "sidecar-injection"
	// SidecarNameAnno is the annotation key for the sidecar name
	SidecarNameAnno = "sidecar.v6d.io/name"
)

// Handle handles admission requests.
func (r *SidecarInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	logger := log.FromContext(ctx).WithName("SidecarInjector")

	pod := &corev1.Pod{}
	if err := r.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	anno := pod.Annotations
	if v, ok := anno[SidecarNameAnno]; ok && v == "default" {
		// create the default sidecar cr
		sidecar := v1alpha1.Sidecar{}
		// get the pod's label as cr's name
		labels := pod.Labels
		keys := []string{}
		for k := range labels {
			if !strings.Contains(k, SidecarEnableLabel) {
				keys = append(keys, k)
			}
		}
		if len(keys) == 0 {
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("the pod doesn't contain a pod selector"))
		}

		sort.Strings(keys)
		selectorname := strings.Join([]string{keys[0], labels[keys[0]]}, "-")
		sidecar.Name = selectorname + "-default-sidecar"
		sidecar.Namespace = pod.Namespace

		err := r.Client.Get(ctx, types.NamespacedName{Name: sidecar.Name, Namespace: sidecar.Namespace}, &sidecar)
		if err != nil && !apierrors.IsNotFound(err) {
			logger.Info("Get sidecar cr failed", "error", err)
			return admission.Errored(http.StatusInternalServerError, err)
		}
		// if the default sidecar cr doesn't exist, create it
		if apierrors.IsNotFound(err) {
			sidecar.Spec.Replicas = 1
			sidecar.Spec.Selector = keys[0] + "=" + labels[keys[0]]
			if err := r.Client.Create(ctx, &sidecar); err != nil {
				logger.Error(err, "failed to create default sidecar cr")
				return admission.Errored(http.StatusInternalServerError, err)
			}
		} else {
			// the default sidecar cr exists, update it
			sidecar.Spec.Replicas += 1
			if err := r.Client.Update(ctx, &sidecar); err != nil {
				logger.Error(err, "failed to update default sidecar cr")
				return admission.Errored(http.StatusInternalServerError, err)
			}
		}
		r.ApplyCRToSidecar(&sidecar, pod)
	}

	marshaledPod, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}

	logger.Info("Injecting vineyard sidecar container successfully!")
	return admission.PatchResponseFromRaw(req.Object.Raw, marshaledPod)
}

func (r *SidecarInjector) BuildVineyardCommand(sidecar *v1alpha1.Sidecar) string {
	socket := "/var/run/vineyard.sock"
	if sidecar.Spec.Volume.PvcName != "" {
		socket = sidecar.Spec.Volume.MountPath + "/vineyard.sock"
	}
	command := "/usr/bin/wait-for-it.sh -t 60 etcd-for-vineyard." + sidecar.Namespace + ".svc.cluster.local:2379;" + "\n" +
		"sleep 1;" + "\n" +
		"/usr/local/bin/vineyardd" + "\n" +
		"--sync_crds true" + "\n" +
		"--socket " + socket + "\n" +
		"--size " + sidecar.Spec.Size + "\n" +
		"--stream_threshold " + strconv.Itoa(int(sidecar.Spec.StreamThreshold)) + "\n" +
		"--etcd_cmd " + sidecar.Spec.EtcdCmd + "\n" +
		"--etcd_prefix " + sidecar.Spec.EtcdPrefix + "\n" +
		"--etcd_endpoint " + sidecar.Spec.EtcdEndpoint + "\n"

	if sidecar.Spec.Spill.Path != "" {
		command = command + "--spill_path " + sidecar.Spec.Spill.Path + "\n" +
			"--spill_lower_rate " + sidecar.Spec.Spill.SpillLowerRate + "\n" +
			"--spill_upper_rate " + sidecar.Spec.Spill.SpillUpperRate + "\n"
	}

	if sidecar.Spec.Metric.Enable {
		command = command + "--metrics" + "\n" +
			"-log_dir /var/log/vineyard/" + "\n"
	}

	return command
}
func (r *SidecarInjector) ApplyCRToSidecar(sidecar *v1alpha1.Sidecar, pod *corev1.Pod) {
	// add sleep to wait for the sidecar container to be ready
	for i := range pod.Spec.Containers {
		pod.Spec.Containers[i].Lifecycle = &corev1.Lifecycle{
			PostStart: &corev1.LifecycleHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"/bin/sh",
						"-c",
						"sleep 120",
					},
				},
			},
		}
	}
	// add the sidecar container
	sidecarContainer := corev1.Container{
		Name:            "vineyard-sidecar",
		Image:           sidecar.Spec.Image,
		ImagePullPolicy: corev1.PullPolicy(sidecar.Spec.ImagePullPolicy),
		Command:         []string{"/bin/bash", "-c", r.BuildVineyardCommand(sidecar)},
		Env: []corev1.EnvVar{
			{
				Name: "VINEYARDD_UID",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "metadata.uid",
					},
				},
			},
			{
				Name: "VINEYARDD_NAME",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "metadata.name",
					},
				},
			},
			{
				Name: "VINEYARDD_NAMESPACE",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "metadata.namespace",
					},
				},
			},
		},
		Ports: []corev1.ContainerPort{
			{
				Name:          "vineyard-rpc",
				ContainerPort: 9600,
				Protocol:      corev1.ProtocolTCP,
			},
		},
	}

	// add the sidecar container
	pod.Spec.Containers = append(pod.Spec.Containers, sidecarContainer)

	// add the metric of sidecar
	if sidecar.Spec.Metric.Enable {
		metricContainer := corev1.Container{
			Name:            "vineyard-metric",
			Image:           sidecar.Spec.Metric.Image,
			ImagePullPolicy: corev1.PullPolicy(sidecar.Spec.Metric.ImagePullPolicy),
			Command:         []string{"/bin/bash", "-c", "./grok_exporter"},
			Args:            []string{"-config", "grok_exporter.yml", "-disable-exporter-metrics", "&"},
			Ports: []corev1.ContainerPort{
				{
					Name:          "exporter",
					ContainerPort: 9144,
					Protocol:      corev1.ProtocolTCP,
				},
			},
			VolumeMounts: []corev1.VolumeMount{
				{
					Name:      "log",
					MountPath: "/var/log/vineyard",
				},
			},
		}
		pod.Spec.Containers = append(pod.Spec.Containers, metricContainer)
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: "log",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})
	}
	// add rpc labels to the pod
	labels := pod.Labels
	s := strings.Split(sidecar.Spec.Service.Selector, "=")
	// add the rpc label selector to the pod's labels
	labels[s[0]] = s[1]

	if sidecar.Spec.Volume.PvcName == "" {
		// add emptyDir volume for sidecar
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: "vineyard-socket",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})

		// add emptyDir volumeMount for every container
		for i := range pod.Spec.Containers {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, corev1.VolumeMount{
				Name:      "vineyard-socket",
				MountPath: "/var/run",
			})
		}
	} else {
		// add pvc volume for sidecar
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: "vineyard-socket",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: sidecar.Spec.Volume.PvcName,
				},
			},
		})
		for i := range pod.Spec.Containers {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, corev1.VolumeMount{
				Name:      "vineyard-socket",
				MountPath: sidecar.Spec.Volume.MountPath,
			})
		}
	}

}

// InjectDecoder injects the decoder.
func (r *SidecarInjector) InjectDecoder(d *admission.Decoder) error {
	r.decoder = d
	return nil
}
