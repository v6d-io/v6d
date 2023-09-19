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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// CSISidecar holds the configuration for the CSI sidecar container
type CSISidecar struct {
	// ProvisionerImage is the image of the provisioner sidecar
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="registry.k8s.io/sig-storage/csi-provisioner:v3.3.0"
	ProvisionerImage string `json:"provisionerImage,omitempty"`

	// AttacherImage is the image of the attacher sidecar
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="registry.k8s.io/sig-storage/csi-attacher:v4.0.0"
	AttacherImage string `json:"attacherImage,omitempty"`

	// NodeRegistrarImage is the image of the node registrar sidecar
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="registry.k8s.io/sig-storage/csi-node-driver-registrar:v2.6.0"
	NodeRegistrarImage string `json:"nodeRegistrarImage,omitempty"`

	// LivenessProbeImage is the image of the liveness probe sidecar
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="registry.k8s.io/sig-storage/livenessprobe:v2.8.0"
	LivenessProbeImage string `json:"livenessProbeImage,omitempty"`

	// ImagePullPolicy is the image pull policy of all sidecar containers
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="Always"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// EnableTopology is the flag to enable topology for the csi driver
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=false
	EnableTopology bool `json:"enableTopology,omitempty"`
}

// VineyardClusters contains the list of vineyard clusters
type VineyardClusters struct {
	// Namespace is the namespace of the vineyard cluster
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=""
	Namespace string `json:"namespace,omitempty"`

	// Name is the name of the vineyard deployment
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=""
	Name string `json:"name,omitempty"`
}

// CSIDriverSpec defines the desired state of CSIDriver
type CSIDriverSpec struct {
	// Image is the name of the csi driver image
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="vineyardcloudnative/vineyard-operator"
	Image string `json:"image,omitempty"`

	// ImagePullPolicy is the image pull policy of the csi driver
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// StorageClassName is the name of the storage class
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="vineyard-csi"
	StorageClassName string `json:"storageClassName,omitempty"`

	// VolumeBindingMode is the volume binding mode of the storage class
	// +kubebuilder:validation:Required
	// +kubebuilder:default:="WaitForFirstConsumer"
	VolumeBindingMode string `json:"volumeBindingMode,omitempty"`

	// Sidecar is the configuration for the CSI sidecar container
	// +kubebuilder:validation:Required
	//nolint: lll
	// +kubebuilder:default:={provisionerImage: "registry.k8s.io/sig-storage/csi-provisioner:v3.3.0", attacherImage: "registry.k8s.io/sig-storage/csi-attacher:v4.0.0", nodeRegistrarImage: "registry.k8s.io/sig-storage/csi-node-driver-registrar:v2.6.0", livenessProbeImage: "registry.k8s.io/sig-storage/livenessprobe:v2.8.0", imagePullPolicy: "Always", enableTopology: false}
	Sidecar CSISidecar `json:"sidecar,omitempty"`

	// Clusters are the list of vineyard clusters
	// +kubebuilder:validation:Required
	// +kubebuilder:default:={}
	Clusters []VineyardClusters `json:"clusters,omitempty"`

	// EnableToleration is the flag to enable toleration for the csi driver
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=false
	EnableToleration bool `json:"enableToleration,omitempty"`

	// EnableVerboseLog is the flag to enable verbose log for the csi driver
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=false
	EnableVerboseLog bool `json:"enableVerboseLog,omitempty"`
}

// CSIDriverStatus defines the observed state of CSIDriver
type CSIDriverStatus struct {
	// State is the state of the csi driver
	State string `json:"state,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:resource:scope=Cluster

// CSIDriver is the Schema for the csidrivers API
type CSIDriver struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CSIDriverSpec   `json:"spec,omitempty"`
	Status CSIDriverStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// CSIDriverList contains a list of CSIDriver
type CSIDriverList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CSIDriver `json:"items"`
}

const (
	// CSIDriverRunning is the running state of the csi driver
	CSIDriverRunning = "running"
	// CSIDriverPending is the pending state of the csi driver
	CSIDriverPending = "pending"
)

func init() {
	SchemeBuilder.Register(&CSIDriver{}, &CSIDriverList{})
}
