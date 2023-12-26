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

// Package sidecar contains the logic for injecting vineyard sidecar
package sidecar

import (
	"bytes"
	"context"
	"encoding/json"
	"html/template"
	"net/http"
	"regexp"
	"sort"
	"strings"

	"github.com/pkg/errors"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
	"sigs.k8s.io/yaml"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/injector"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// nolint: lll
// +kubebuilder:webhook:admissionReviewVersions=v1,sideEffects=None,path=/mutate-v1-pod-sidecar,mutating=true,failurePolicy=fail,groups="",resources=pods,verbs=create;update,versions=v1,name=mpod.sidecar.kb.io

// Injector injects vineyard sidecar container into Pods
type Injector struct {
	client.Client
	decoder *admission.Decoder
}

// Handle handles admission requests.
func (r *Injector) Handle(ctx context.Context, req admission.Request) admission.Response {
	logger := log.FromContext(ctx).WithName("Injector")

	sidecar := &v1alpha1.Sidecar{}
	templatePod := &corev1.Pod{}
	pod := &corev1.Pod{}
	if err := r.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	anno := pod.Annotations
	if v, ok := anno[annotations.SidecarNameAnno]; ok && v == "default" {
		// get the pod's label as cr's name
		l := pod.Labels
		keys := []string{}
		for k := range l {
			if !strings.Contains(k, labels.SidecarEnableLabel) {
				keys = append(keys, k)
			}
		}
		if len(keys) == 0 {
			return admission.Errored(
				http.StatusInternalServerError,
				errors.Errorf("the pod doesn't contain a pod selector"),
			)
		}

		sort.Strings(keys)
		selectorname := strings.Join([]string{keys[0], l[keys[0]]}, "-")
		if len(selectorname) > 63-len("-default-sidecar") {
			selectorname = selectorname[:63-len("-default-sidecar")]
		}
		sidecar.Name = selectorname + "-default-sidecar"
		// replace the invalid characters in the name
		re := regexp.MustCompile(`[^a-zA-Z0-9.-]+`)
		sidecar.Name = re.ReplaceAllString(sidecar.Name, "-")
		sidecar.Namespace = pod.Namespace
		sidecar.OwnerReferences = pod.OwnerReferences
		err := r.Get(
			ctx,
			types.NamespacedName{Name: sidecar.Name, Namespace: sidecar.Namespace},
			sidecar,
		)
		if err != nil && !apierrors.IsNotFound(err) {
			logger.Info("Get sidecar cr failed", "error", err)
			return admission.Errored(http.StatusInternalServerError, err)
		}
		// if the default sidecar cr doesn't exist, create it
		if apierrors.IsNotFound(err) {
			sidecar.Spec.Replicas = 1
			// use default configurations
			sidecar.Spec.Selector = keys[0] + "=" + l[keys[0]]

			if err := r.Create(ctx, sidecar); err != nil {
				logger.Error(err, "failed to create default sidecar cr")
				return admission.Errored(http.StatusInternalServerError, err)
			}
		} else {
			if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
				name := client.ObjectKeyFromObject(sidecar)
				if err := r.Get(ctx, name, sidecar); err != nil {
					return errors.Wrap(err, "failed to get "+name.String())
				}
				// the default sidecar cr exists, update it
				sidecar.Spec.Replicas++
				if err := r.Update(ctx, sidecar); err != nil {
					return errors.Wrap(err, "failed to update sidecar's status")
				}
				return nil
			}); err != nil {
				logger.Error(err, "failed to update default sidecar cr")
				return admission.Errored(http.StatusInternalServerError, err)
			}
		}
	} else {
		// get the sidecar cr
		if err := r.Get(
			ctx,
			types.NamespacedName{Name: v, Namespace: pod.Namespace},
			sidecar,
		); err != nil {
			logger.Error(err, "get custom sidecar cr failed")
			return admission.Errored(http.StatusInternalServerError, err)
		}
	}
	buf, err := templates.ReadFile("sidecar/injection-template.yaml")
	if err != nil {
		logger.Error(err, "failed to read injection template")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	tmplFunc := map[string]interface{}{
		"toYaml": func(v interface{}) string {
			bs, err := yaml.Marshal(v)
			if err != nil {
				logger.Error(err, "failed to marshal object %v to yaml", v)
				return ""
			}
			return string(bs)
		},
		"indent": func(spaces int, s string) string {
			prefix := strings.Repeat(" ", spaces)
			return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
		},
	}

	if tpl, err := template.New("sidecar").Funcs(tmplFunc).Parse(string(buf)); err == nil {
		var buf bytes.Buffer
		if err := tpl.Execute(&buf, sidecar); err == nil {
			decode := scheme.Codecs.UniversalDeserializer().Decode
			obj, _, _ := decode(buf.Bytes(), nil, nil)
			templatePod = obj.(*corev1.Pod)
		} else {
			logger.Error(err, "failed to execute template")
			return admission.Errored(http.StatusInternalServerError, err)
		}
	}
	if err := r.ApplyToSidecar(sidecar, templatePod, pod, true); err != nil {
		logger.Error(err, "failed to apply sidecar cr to pod")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	marshaledPod, err := json.Marshal(pod)
	if err != nil {
		logger.Error(err, "failed to marshal pod")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	logger.Info("Injecting vineyard sidecar container successfully!")
	return admission.PatchResponseFromRaw(req.Object.Raw, marshaledPod)
}

// ApplyToSidecar applies the sidecar cr and pod to the sidecar
func (r *Injector) ApplyToSidecar(
	sidecar *v1alpha1.Sidecar,
	pod *corev1.Pod,
	podWithSidecar *corev1.Pod,
	addLabels bool,
) error {
	// Convert the podWithSidecar object to an unstructured object.
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(podWithSidecar)
	if err != nil {
		return errors.Wrap(err, "failed to convert podWithSidecar to unstructured object")
	}
	unstructuredPodWithSidecar := &unstructured.Unstructured{Object: obj}

	// Convert the Pod object to an unstructured object.
	obj, err = runtime.DefaultUnstructuredConverter.ToUnstructured(pod)
	if err != nil {
		return errors.Wrap(err, "failed to convert pod to unstructured object")
	}
	unstructuredPod := &unstructured.Unstructured{Object: obj}

	selector := ""
	if addLabels {
		selector = "rpc.vineyardd.v6d.io/rpc=vineyardd-rpc"
	}
	if err := injector.InjectSidecar(unstructuredPodWithSidecar, unstructuredPod, sidecar, selector); err != nil {
		return errors.Wrap(err, "failed to inject sidecar")
	}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(
		unstructuredPodWithSidecar.Object, podWithSidecar); err != nil {
		return errors.Wrap(err, "failed to convert unstructured object to podWithSidecar")
	}

	return nil
}

// InjectDecoder injects the decoder.
func (r *Injector) InjectDecoder(d *admission.Decoder) error {
	r.decoder = d
	return nil
}
