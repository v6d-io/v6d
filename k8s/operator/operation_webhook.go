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

// Package operator contains the logic for the operator
package operator

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/v6d-io/v6d/k8s/schedulers"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// log is for logging in this package.
var assemblyInjectorLog = logf.Log.WithName("assembly_injector")

// nolint: lll
// +kubebuilder:webhook:admissionReviewVersions=v1,sideEffects=None,path=/mutate-v1-pod,mutating=true,failurePolicy=fail,groups="",resources=pods,verbs=create;update,versions=v1,name=mpod.kb.io

// AssemblyInjector injects assembly operation container into Pods
type AssemblyInjector struct {
	Client  client.Client
	decoder *admission.Decoder
}

const (
	// AssmeblyEnabledLabel is the label for assembly, and inject the assembly container when setting true
	AssmeblyEnabledLabel = "assembly.v6d.io/enabled"
	// RepartitionEnabledLabel is the label for repartition, and inject the repartition container when setting true
	RepartitionEnabledLabel = "repartition.v6d.io/enabled"
)

// LabelRequiredPods labels the pods with the given label
func (r *AssemblyInjector) LabelRequiredPods(ctx context.Context, pod *corev1.Pod, label string) error {
	if value, ok := pod.Labels[label]; ok && strings.ToLower(value) == "true" {
		if requiredJob, ok := pod.Annotations[schedulers.VineyardJobRequired]; ok {
			jobs := strings.Split(requiredJob, ".")
			for _, job := range jobs {
				// get the required job
				podList := &corev1.PodList{}
				opts := []client.ListOption{
					client.MatchingLabels{
						"app": job,
					},
				}
				if err := r.Client.List(ctx, podList, opts...); err != nil {
					return fmt.Errorf("Failed to list pods: %v", err)
				}
				for i := range podList.Items {
					// label the required pods that need to be injected with the assembly container
					labels := &podList.Items[i].Labels
					(*labels)["need-injected-"+label[:strings.Index(label, ".")]] = "true"
					if err := r.Client.Update(ctx, &podList.Items[i], &client.UpdateOptions{}); err != nil {
						return fmt.Errorf("Failed to update pod: %v", err)
					}
				}
			}
		}
	}
	return nil
}

// Handle handles admission requests.
func (r *AssemblyInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := r.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	operationLabels := []string{AssmeblyEnabledLabel, RepartitionEnabledLabel}
	// check all operation labels
	for _, l := range operationLabels {
		if err := r.LabelRequiredPods(ctx, pod, l); err != nil {
			return admission.Errored(http.StatusBadRequest, fmt.Errorf("assembly label error: %v", err))
		}
	}

	// Add podname and podnamespace to all pods' env
	for i := range pod.Spec.Containers {
		m := map[string]bool{}
		for _, e := range pod.Spec.Containers[i].Env {
			m[e.Name] = true
		}
		if _, ok := m["POD_NAME"]; !ok {
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name: "POD_NAME",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "metadata.name",
					},
				},
			})
		}
		if _, ok := m["POD_NAMESPACE"]; !ok {
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name: "POD_NAMESPACE",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "metadata.namespace",
					},
				},
			})
		}
	}

	marshaledPod, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}

	assemblyInjectorLog.Info("Injecting the env and assembly labels successfully!")
	return admission.PatchResponseFromRaw(req.Object.Raw, marshaledPod)
}

// InjectDecoder injects the decoder.
func (r *AssemblyInjector) InjectDecoder(d *admission.Decoder) error {
	r.decoder = d
	return nil
}
