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

// Package injector contains the logic for injecting the vineyard sidecar into a pod.
package injector

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"github.com/pkg/errors"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
)

var (
	// PodKind is the kind of the kubernetes pod.
	PodKind = "Pod"
)

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
		labels, _, err = unstructured.NestedStringMap(obj.Object, "spec", "template", "metadata", "labels")
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
		err = unstructured.SetNestedStringMap(obj.Object, labels, "spec", "template", "metadata", "labels")
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
		containers, _, err = unstructured.NestedSlice(obj.Object, "spec", "template", "spec", "containers")
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
		err = unstructured.SetNestedSlice(obj.Object, containers, "spec", "template", "spec", "containers")
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
		volumes, _, err = unstructured.NestedSlice(obj.Object, "spec", "template", "spec", "volumes")
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
		err = unstructured.SetNestedSlice(obj.Object, volumes, "spec", "template", "spec", "volumes")
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

// injectContainersAndVolumes injects the sidecar containers and volumes into the workload containers and volumes.
func injectContainersAndVolumes(workloadContainers []interface{},
	workloadVolumes []interface{},
	sidecarContainers []interface{},
	sidecarVolumes []interface{},
	sidecar *v1alpha1.Sidecar,
) ([]interface{}, []interface{}) {
	var containers []interface{}
	var volumes []interface{}

	mountPath := "/var/run"
	pvcName := sidecar.Spec.Volume.PvcName
	if pvcName != "" || sidecar.Spec.Volume.MountPath != "" {
		mountPath = sidecar.Spec.Volume.MountPath
	}

	// sleepCommand is the command to wait for the sidecar container to be ready.
	sleepCommand := fmt.Sprintf("while [ ! -e %s/vineyard.sock ]; do sleep 1; done;", mountPath)

	// add sleep to wait for the sidecar container to be ready
	for i := range workloadContainers {
		c := workloadContainers[i].(map[string]interface{})
		var commands []interface{}
		// for the nil command, we can't add the sleep command
		// because the added sleep command will overlay the entrypoint
		// of the container.
		// TODO: wait for the sidecar container to be ready and not overlay the entrypoint.
		if c["command"] != nil {
			commands = c["command"].([]interface{})
			commands[len(commands)-1] = fmt.Sprintf("%s%s", sleepCommand, commands[len(commands)-1])
		}
		c["command"] = commands
	}

	for i := range workloadContainers {
		c := workloadContainers[i].(map[string]interface{})
		var volumeMounts []interface{}
		vm := map[string]interface{}{
			"name":      "vineyard-socket",
			"mountPath": mountPath,
		}
		if c["volumeMounts"] == nil {
			volumeMounts = []interface{}{vm}
		} else {
			volumeMounts = c["volumeMounts"].([]interface{})
			volumeMounts = append(volumeMounts, vm)
		}
		c["volumeMounts"] = volumeMounts
	}

	containers = append(workloadContainers, sidecarContainers...)
	volumes = append(workloadVolumes, sidecarVolumes...)

	return containers, volumes
}

// InjectSidecar injects the sidecar into the given unstructured kubernetes object.
func InjectSidecar(workload, sidecar *unstructured.Unstructured, s *v1alpha1.Sidecar, selector string) error {
	workloadContainers, workloadVolumes, err := GetContainersAndVolumes(workload)
	if err != nil {
		return errors.Wrap(err, "failed to get containers and volumes from workload")
	}

	sidecarContainers, sidecarVolumes, err := GetContainersAndVolumes(sidecar)
	if err != nil {
		return errors.Wrap(err, "failed to get containers and volumes from sidecar")
	}

	containers, volumes := injectContainersAndVolumes(workloadContainers,
		workloadVolumes, sidecarContainers,
		sidecarVolumes, s)

	err = SetContainer(workload, containers)
	if err != nil {
		return err
	}

	err = SetVolume(workload, volumes)
	if err != nil {
		return err
	}

	// add selector to the workload
	if selector != "" {
		labels, err := GetLabels(workload)
		if err != nil {
			return err
		}
		s := strings.Split(selector, "=")
		// add the rpc label selector to the workload
		labels[s[0]] = s[1]
		err = SetLabels(workload, labels)
		if err != nil {
			return err
		}
	}
	return nil
}
