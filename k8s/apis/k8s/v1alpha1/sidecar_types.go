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

// Package v1alpha1 contains API Schema definitions for the k8s v1alpha1 API group
package v1alpha1

import (
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

	// vineyard container configuration
	// +kububuilder:validation:Optional
	//nolint: lll
	// +kubebuilder:default:={image: "vineyardcloudnative/vineyardd:latest", imagePullPolicy: "IfNotPresent", syncCRDs: true, socket: "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}", size: "256Mi", streamThreshold: 80, etcdEndpoint: "http://etcd-for-vineyard:2379", etcdPrefix: "/vineyard"}
	VineyardConfig VineyardContainerConfig `json:"vineyardConfig,omitempty"`

	// metric container configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={enable: false, image: "vineyardcloudnative/vineyard-grok-exporter:latest", imagePullPolicy: "IfNotPresent"}
	MetricConfig MetricContainerConfig `json:"metricConfig,omitempty"`

	// metric configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Volume VolumeConfig `json:"volume,omitempty"`

	// rpc service configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={type: "ClusterIP", port: 9600, selector: "rpc.vineyardd.v6d.io/rpc=vineyard-rpc"}
	Service ServiceConfig `json:"service,omitempty"`
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

// Sidecar is the Schema for the sidecars API
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
