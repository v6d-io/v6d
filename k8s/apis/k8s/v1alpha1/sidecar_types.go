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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SidecarSpec defines the desired state of Sidecar
type SidecarSpec struct {
	// the selector of pod
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	Selector string `json:"selector,omitempty"`

	// the replicas of workload
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=0
	Replicas int `json:"replicas,omitempty"`

	// EtcdReplicas describe the etcd replicas
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=1
	EtcdReplicas int `json:"etcdReplicas,omitempty"`

	// vineyard container configuration
	// +kububuilder:validation:Optional
	//nolint: lll
	// +kubebuilder:default:={image: "vineyardcloudnative/vineyardd:latest", imagePullPolicy: "IfNotPresent", syncCRDs: true, socket: "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}", size: "", streamThreshold: 80}
	Vineyard VineyardConfig `json:"vineyard,omitempty"`

	// metric container configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={enable: false, image: "vineyardcloudnative/vineyard-grok-exporter:latest", imagePullPolicy: "IfNotPresent"}
	Metric MetricConfig `json:"metric,omitempty"`

	// metric configurations
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Volume VolumeConfig `json:"volume,omitempty"`

	// rpc service configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={type: "ClusterIP", port: 9600}
	Service ServiceConfig `json:"service,omitempty"`

	// SecurityContext holds the security context settings for the vineyardd container.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	SecurityContext corev1.SecurityContext `json:"securityContext,omitempty"`

	// Volumes is the list of Kubernetes volumes that can be mounted by the vineyard container.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// VolumeMounts specifies the volumes listed in ".spec.volumes" to mount into the vineyard container.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`
}

// SidecarStatus defines the observed state of Sidecar
type SidecarStatus struct {
	// the replicas of injected sidecar
	// +kubebuilder:validation:Optional
	Current int32 `json:"current,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Current",type=string,JSONPath=`.status.current`
// +kubebuilder:printcolumn:name="Desired",type=string,JSONPath=`.spec.replicas`

// Sidecar is used for configuring and managing the vineyard sidecar container.
type Sidecar struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SidecarSpec   `json:"spec,omitempty"`
	Status SidecarStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// SidecarList contains a list of Sidecar
type SidecarList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Sidecar `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Sidecar{}, &SidecarList{})
}
