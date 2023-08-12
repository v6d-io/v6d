/*
* Copyright 2020-2023 Alibaba Group Holding Limited.

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
package util

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

const (
	// PodKind is the kind of the kubernetes pod.
	PodKind = "Pod"
)

// GetName returns the name of the given unstructured kubernetes object.
func GetName(obj *unstructured.Unstructured) (string, error) {
	name, _, err := unstructured.NestedString(obj.Object, "metadata", "name")
	if err != nil {
		return "", err
	}
	return name, nil
}

// GetNamespace returns the namespace of the given unstructured kubernetes object.
func GetNamespace(obj *unstructured.Unstructured) (string, error) {
	namespace, _, err := unstructured.NestedString(obj.Object, "metadata", "namespace")
	if err != nil {
		return "", err
	}
	return namespace, nil
}

// GetAnnotations returns the annotations of the given unstructured kubernetes object.
func GetAnnotations(obj *unstructured.Unstructured) (map[string]string, error) {
	kind := obj.GetKind()
	var (
		annotations map[string]string
		err         error
	)
	if kind == PodKind {
		annotations, _, err = unstructured.NestedStringMap(obj.Object, "metadata",
			"annotations")
	} else {
		annotations, _, err = unstructured.NestedStringMap(obj.Object, "spec",
			"template", "metadata", "annotations")
	}
	if err != nil {
		return nil, err
	}
	return annotations, nil
}

// SetAnnotations sets the annotations of the given unstructured kubernetes object.
func SetAnnotations(obj *unstructured.Unstructured, annotations map[string]string) error {
	kind := obj.GetKind()
	var err error
	if kind == PodKind {
		err = unstructured.SetNestedStringMap(obj.Object, annotations, "metadata",
			"annotations")
	} else {
		err = unstructured.SetNestedStringMap(obj.Object, annotations, "spec",
			"template", "metadata", "annotations")
	}
	if err != nil {
		return err
	}
	return nil
}

// GetLabels returns the labels of the given unstructured kubernetes object.
func GetLabels(obj *unstructured.Unstructured) (map[string]string, error) {
	kind := obj.GetKind()
	var (
		labels map[string]string
		err    error
	)
	if kind == PodKind {
		labels, _, err = unstructured.NestedStringMap(obj.Object, "metadata", "labels")
	} else {
		labels, _, err = unstructured.NestedStringMap(obj.Object, "spec", "template",
			"metadata", "labels")
	}
	if err != nil {
		return nil, err
	}
	return labels, nil
}

// SetLabels sets the labels of the given unstructured kubernetes object.
func SetLabels(obj *unstructured.Unstructured, labels map[string]string) error {
	kind := obj.GetKind()
	var err error
	if kind == PodKind {
		err = unstructured.SetNestedStringMap(obj.Object, labels, "metadata", "labels")
	} else {
		err = unstructured.SetNestedStringMap(obj.Object, labels, "spec", "template",
			"metadata", "labels")
	}
	if err != nil {
		return err
	}
	return nil
}

// GetContainer returns the container interface with the given unstructured kubernetes object.
func GetContainer(obj *unstructured.Unstructured) ([]interface{}, error) {
	kind := obj.GetKind()
	var (
		containers []interface{}
		err        error
	)
	if kind == PodKind {
		containers, _, err = unstructured.NestedSlice(obj.Object, "spec", "containers")
	} else {
		containers, _, err = unstructured.NestedSlice(obj.Object, "spec", "template",
			"spec", "containers")
	}
	if err != nil {
		return nil, err
	}
	return containers, nil
}

// SetContainer sets the container with the given unstructured kubernetes object.
func SetContainer(obj *unstructured.Unstructured, containers []interface{}) error {
	kind := obj.GetKind()
	var err error
	if kind == PodKind {
		err = unstructured.SetNestedSlice(obj.Object, containers, "spec", "containers")
	} else {
		err = unstructured.SetNestedSlice(obj.Object, containers, "spec", "template",
			"spec", "containers")
	}
	if err != nil {
		return err
	}
	return nil
}

// GetVolume returns the volume interface with the given unstructured kubernetes object.
func GetVolume(obj *unstructured.Unstructured) ([]interface{}, error) {
	kind := obj.GetKind()
	var (
		volumes []interface{}
		err     error
	)
	if kind == PodKind {
		volumes, _, err = unstructured.NestedSlice(obj.Object, "spec", "volumes")
	} else {
		volumes, _, err = unstructured.NestedSlice(obj.Object, "spec", "template",
			"spec", "volumes")
	}
	if err != nil {
		return nil, err
	}
	return volumes, nil
}

// SetVolume sets the volume with the given unstructured kubernetes object.
func SetVolume(obj *unstructured.Unstructured, volumes []interface{}) error {
	kind := obj.GetKind()
	var err error
	if kind == PodKind {
		err = unstructured.SetNestedSlice(obj.Object, volumes, "spec", "volumes")
	} else {
		err = unstructured.SetNestedSlice(obj.Object, volumes, "spec", "template",
			"spec", "volumes")
	}

	if err != nil {
		return err
	}
	return nil
}

// GetContainersAndVolumes returns the containers and volumes of the given unstructured kubernetes object.
func GetContainersAndVolumes(obj *unstructured.Unstructured) ([]interface{}, []interface{}, error) {
	containers, err := GetContainer(obj)
	if err != nil {
		return nil, nil, err
	}
	volumes, err := GetVolume(obj)
	if err != nil {
		return nil, nil, err
	}
	return containers, volumes, nil
}

// GetRequiredPodAffinity returns the required pod affinity of the given unstructured kubernetes object.
func GetRequiredPodAffinity(obj *unstructured.Unstructured) ([]interface{}, error) {
	kind := obj.GetKind()
	var (
		required []interface{}
		err      error
	)
	if kind == PodKind {
		required, _, err = unstructured.NestedSlice(obj.Object, "spec", "affinity",
			"podAffinity")
	} else {
		required, _, err = unstructured.NestedSlice(obj.Object, "spec", "template",
			"spec", "affinity", "podAffinity")
	}

	if err != nil {
		return nil, err
	}
	return required, nil
}

// SetRequiredPodAffinity sets the required pod affinity of the given unstructured kubernetes object.
func SetRequiredPodAffinity(obj *unstructured.Unstructured, required []interface{}) error {
	kind := obj.GetKind()
	var err error
	if kind == PodKind {
		err = unstructured.SetNestedSlice(obj.Object, required, "spec", "affinity",
			"podAffinity", "requiredDuringSchedulingIgnoredDuringExecution")
	} else {
		err = unstructured.SetNestedSlice(obj.Object, required, "spec", "template", "spec",
			"affinity", "podAffinity", "requiredDuringSchedulingIgnoredDuringExecution")
	}
	if err != nil {
		return err
	}
	return nil
}

// GetStatus returns the status of the given unstructured kubernetes object.
func GetStatus(obj *unstructured.Unstructured) (map[string]interface{}, error) {
	status, _, err := unstructured.NestedMap(obj.Object, "status")
	if err != nil {
		return nil, err
	}
	return status, nil
}

// SetNodename sets the nodename of the given unstructured kubernetes object.
func SetNodename(obj *unstructured.Unstructured, nodename string) error {
	kind := obj.GetKind()
	var err error
	if kind == PodKind {
		err = unstructured.SetNestedField(obj.Object, nodename, "spec", "nodeName")
	} else {
		err = unstructured.SetNestedField(obj.Object, nodename, "spec", "template",
			"spec", "nodeName")
	}

	if err != nil {
		return err
	}
	return nil
}
