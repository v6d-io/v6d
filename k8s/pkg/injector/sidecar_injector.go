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
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var (
	// SocketEnv is the env of vineyard ipc socket.
	SocketEnv = "VINEYARD_IPC_SOCKET"
)

// InjectEnv injects the vineyard ipc socket env into the containers env.
func InjectEnv(containers *[]interface{}, mountPath string) {
	// check if the vineyard ipc socket env is already exist
	// if exist, update the value
	for _, c := range *containers {
		if c.(map[string]interface{})["env"] == nil {
			c.(map[string]interface{})["env"] = make([]interface{}, 0)
		}
		env := c.(map[string]interface{})["env"].([]interface{})
		newEnv := make([]interface{}, 0)
		for _, e := range env {
			if e.(map[string]interface{})["name"] == SocketEnv {
				continue
			}
			newEnv = append(newEnv, e)
		}
		newEnv = append(newEnv, map[string]interface{}{
			"name":  SocketEnv,
			"value": mountPath + "/vineyard.sock",
		})
		c.(map[string]interface{})["env"] = newEnv
	}
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

	InjectEnv(&workloadContainers, mountPath)

	containers = append(workloadContainers, sidecarContainers...)
	volumes = append(workloadVolumes, sidecarVolumes...)

	return containers, volumes
}

// InjectSidecar injects the sidecar into the given unstructured kubernetes object.
func InjectSidecar(workload, sidecar *unstructured.Unstructured, s *v1alpha1.Sidecar, selector string) error {
	workloadContainers, workloadVolumes, err := util.GetContainersAndVolumes(workload)
	if err != nil {
		return errors.Wrap(err, "failed to get containers and volumes from workload")
	}

	sidecarContainers, sidecarVolumes, err := util.GetContainersAndVolumes(sidecar)
	if err != nil {
		return errors.Wrap(err, "failed to get containers and volumes from sidecar")
	}

	containers, volumes := injectContainersAndVolumes(workloadContainers,
		workloadVolumes, sidecarContainers,
		sidecarVolumes, s)

	err = util.SetContainer(workload, containers)
	if err != nil {
		return err
	}

	err = util.SetVolume(workload, volumes)
	if err != nil {
		return err
	}

	labels, err := util.GetLabels(workload)
	if err != nil {
		return err
	}

	if labels == nil {
		labels = make(map[string]string)
	}
	selectors := strings.Split(selector, ",")
	for i := range selectors {
		s := strings.Split(selectors[i], "=")
		// add the rpc label selector to the workload
		labels[s[0]] = s[1]
	}
	err = util.SetLabels(workload, labels)
	if err != nil {
		return err
	}

	return nil
}
