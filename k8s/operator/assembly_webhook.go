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

package operator

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

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
	// AssmeblyEnabledAnnotation is the annotation for assembly, and inject the assembly container when setting true
	AssmeblyEnabledAnnotation = "assembly.v6d.io/enabled"
	// ImageAnnotation is the annotation key for the image to use for the sidecar.
	ImageAnnotation = "assembly.v6d.io/image"
	// CommandAnnotation is the annotation key for the command to run in the assembly container.
	CommandAnnotation = "assembly.v6d.io/command"
	// SharedVolumePathInAssemblyAnnotation is the annotation key for the shared volume path in the assembly container.
	SharedVolumePathInAssemblyAnnotation = "assembly.v6d.io/assembly.shared-volume-path"
	// SharedVolumePathInJobAnnotation is the annotation key for the shared volume path in the job pod.
	SharedVolumePathInJobAnnotation = "assembly.v6d.io/job.shared-volume-path"
)

// AssemblyConfig holds the configuration for the assembly sidecar
type AssemblyConfig struct {
	// Image is the container image name
	Image string
	// Command is the command to run in the assembly container
	Command string
	// SharedVolumePathInAssembly is the path to the shared volume in the assembly container
	SharedVolumePathInAssembly string
	// SharedVolumePathInJob is the path to the shared volume in the job container
	SharedVolumePathInJob string
}

func (r *AssemblyInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := r.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	if pod.Annotations == nil {
		return admission.Allowed("there are no annatations")
	}

	// get all configurations from the annotations of the pod
	annotations := pod.Annotations
	assemblyConfig := AssemblyConfig{}
	if enabled, ok := annotations[AssmeblyEnabledAnnotation]; !ok || enabled == "false" {
		return admission.Allowed("assembly is not enabled")
	}

	if image, ok := annotations[ImageAnnotation]; ok {
		assemblyConfig.Image = image
	} else {
		assemblyInjectorLog.Info("assembly image not found")
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("assembly image not found"))
	}

	if command, ok := annotations[CommandAnnotation]; ok {
		assemblyConfig.Command = command
	} else {
		assemblyInjectorLog.Info("assembly command not found")
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("assembly command not found"))
	}

	if sharedVolumeInAssemblyPath, ok := annotations[SharedVolumePathInAssemblyAnnotation]; ok {
		assemblyConfig.SharedVolumePathInAssembly = sharedVolumeInAssemblyPath
	} else {
		assemblyInjectorLog.Info("assembly shared volume path not found")
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("assembly shared volume path not found"))
	}

	if sharedVolumeInJobPath, ok := annotations[SharedVolumePathInJobAnnotation]; ok {
		assemblyConfig.SharedVolumePathInJob = sharedVolumeInJobPath
	} else {
		assemblyInjectorLog.Info("job shared volume path not found")
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("job shared volume path not found"))
	}

	assemblyInjectorLog.Info("Producing the assemblySidecar container and shared volume...")
	logPath := assemblyConfig.SharedVolumePathInAssembly + "/log"
	assemblySidecar := corev1.Container{
		Name:            "assembly-sidecar",
		Image:           assemblyConfig.Image,
		ImagePullPolicy: corev1.PullIfNotPresent,
		// TODO: Synchronize the two containers.
		Lifecycle: &corev1.Lifecycle{
			PostStart: &corev1.LifecycleHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"sh",
						"-c",
						"while [ -z $(cat " + logPath + " | grep \"stream done\")]; do sleep 0.1; done",
					},
				},
			},
		},
		Command: []string{
			"sh",
			"-c",
			assemblyConfig.Command,
		},
		VolumeMounts: []corev1.VolumeMount{
			corev1.VolumeMount{
				Name:      "shared-volume",
				MountPath: assemblyConfig.SharedVolumePathInAssembly,
			},
		},
	}

	sharedVolume := corev1.Volume{
		Name: "shared-volume",
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}

	assemblyInjectorLog.Info("Injecting the assemblySidecar container and shared volume...")
	// inject the assembly sidecar into the pod
	pod.Spec.Containers = append(pod.Spec.Containers, assemblySidecar)
	pod.Spec.Volumes = append(pod.Spec.Volumes, sharedVolume)

	marshaledPod, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}

	assemblyInjectorLog.Info("Injecting the assemblySidecar container and shared volume successfully!")
	return admission.PatchResponseFromRaw(req.Object.Raw, marshaledPod)
}

// InjectDecoder injects the decoder.
func (r *AssemblyInjector) InjectDecoder(d *admission.Decoder) error {
	r.decoder = d
	return nil
}
