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
	corev1 "k8s.io/api/core/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
)

// SleepCommand is the command to wait for the sidecar container to be ready.
var SleepCommand = "while [ ! -e /var/run/vineyard.sock ]; do sleep 1; done;"

// InjectSidecar injects the vineyard sidecar into the containers of workload.
func InjectSidecar(workloadContainers *[]corev1.Container,
	workloadVolumes *[]corev1.Volume,
	sidecarContainers *[]corev1.Container,
	sidecarVolumes *[]corev1.Volume,
	sidecar *v1alpha1.Sidecar,
) {
	// add sleep to wait for the sidecar container to be ready
	for i := range *workloadContainers {
		if (*workloadContainers)[i].Command == nil {
			(*workloadContainers)[i].Command = []string{"/bin/sh", "-c"}
		}
		(*workloadContainers)[i].Command = append((*workloadContainers)[i].Command,
			SleepCommand)
	}

	pvcName := sidecar.Spec.Volume.PvcName
	mountPath := "/var/run"

	if pvcName != "" {
		mountPath = sidecar.Spec.Volume.MountPath
	}

	for i := range *workloadContainers {
		// add volumeMounts to the app container
		(*workloadContainers)[i].VolumeMounts = append(
			(*workloadContainers)[i].VolumeMounts,
			corev1.VolumeMount{
				Name:      "vineyard-socket",
				MountPath: mountPath,
			},
		)
	}

	// add the sidecar container
	*workloadContainers = append(*workloadContainers, (*sidecarContainers)...)
	// add the sidecar volume
	*workloadVolumes = append(*workloadVolumes, (*sidecarVolumes)...)
}
