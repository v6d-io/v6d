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

// Package v1alpha1 contains API Schema definitions for the k8s v1alpha1 API group
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Spill holds all configuration about spilling
type Spill struct {
	// the path of spilling
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	Path string `json:"path,omitempty"`

	// low watermark of spilling memory
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="0.3"
	SpillLowerRate string `json:"spillLowerRate,omitempty"`

	// high watermark of triggering spilling
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="0.8"
	SpillUpperRate string `json:"spillUpperRate,omitempty"`
}

// Metric holds all metrics configuration
type Metric struct {
	// Enable metrics
	// +kubebuilder:validation:Optional
	// +kubebuilder:default=false
	Enable bool `json:"enable,omitempty"`

	// represent the metric's image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default="vineyardcloudnative/vineyard-grok-exporter:latest"
	Image string `json:"image,omitempty"`

	// the policy about pulling image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`
}

// VolumeConfig holds all configuration about persistent volume
type VolumeConfig struct {
	// the name of pvc
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	PvcName string `json:"pvcName,omitempty"`

	// the mount path of pv
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	MountPath string `json:"mountPath,omitempty"`
}

// SidecarSpec defines the desired state of Sidecar
type SidecarSpec struct {
	// the image of vineyard sidecar image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="vineyardcloudnative/vineyardd:latest"
	Image string `json:"image,omitempty"`

	// the policy about pulling vineyard sidecar image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// the selector of pod
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	Selector string `json:"selector,omitempty"`

	// the replicas of workload
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=0
	Replicas int `json:"replicas,omitempty"`

	// shared memory size for vineyardd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="256Mi"
	Size string `json:"size,omitempty"`

	// The directory on host for the IPC socket file. The UNIX-domain
	// socket will be placed as `${Socket}/vineyard.sock`.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="/var/run"
	Socket string `json:"socket,omitempty"`

	// memory threshold of streams (percentage of total memory)
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=80
	StreamThreshold int64 `json:"streamThreshold,omitempty"`

	// path of etcd executable
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="etcd"
	EtcdCmd string `json:"etcdCmd,omitempty"`

	// endpoint of etcd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="http://etcd-for-vineyard:2379"
	EtcdEndpoint string `json:"etcdEndpoint,omitempty"`

	// path prefix in etcd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="/vineyard"
	EtcdPrefix string `json:"etcdPrefix,omitempty"`

	// the configuration of spilling
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Spill Spill `json:"spill,omitempty"`

	// metric configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Metric Metric `json:"metric,omitempty"`

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
