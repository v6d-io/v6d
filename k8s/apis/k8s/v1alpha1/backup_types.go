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

// BackupSpec defines the desired state of Backup
type BackupSpec struct {
	// the name of the vineyard cluster
	// +kubebuilder:validation:Required
	VineyarddName string `json:"vineyarddName,omitempty"`

	// the namespace of the vineyard cluster
	// +kubebuilder:validation:Required
	VineyarddNamespace string `json:"vineyarddNamespace,omitempty"`

	// the specific objects to be backed up
	// if not specified, all objects will be backed up
	// +kubebuilder:validation:Required
	ObjectIDs []string `json:"objecIDs,omitempty"`

	// the path of backup data
	// +kubebuilder:validation:Required
	BackupPath string `json:"backupPath,omitempty"`

	// the PersistentVolumeSpec of the backup data
	// +kubebuilder:validation:Optional
	PersistentVolumeSpec corev1.PersistentVolumeSpec `json:"persistentVolumeSpec,omitempty"`

	// the PersistentVolumeClaimSpec of the backup data
	// +kubebuilder:validation:Optional
	PersistentVolumeClaimSpec corev1.PersistentVolumeClaimSpec `json:"persistentVolumeClaimSpec,omitempty"`
}

// BackupStatus defines the observed state of Backup
type BackupStatus struct {
	// state of the backup
	// +kubebuilder:validation:Optional
	State string `json:"state,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.state`

// Backup describes a backup operation of vineyard objects, which uses the
// [Kubernetes PersistentVolume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
// to store the backup data. Every backup operation will be binded with the
// name of Backup.
type Backup struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   BackupSpec   `json:"spec,omitempty"`
	Status BackupStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// BackupList contains a list of Backup
type BackupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Backup `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Backup{}, &BackupList{})
}
