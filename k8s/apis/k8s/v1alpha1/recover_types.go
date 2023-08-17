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

// RecoverSpec defines the desired state of Recover
type RecoverSpec struct {
	// the name of backup
	// +kubebuilder:validation:Required
	BackupName string `json:"backupName,omitempty"`

	// the namespace of backup
	// +kubebuilder:validation:Required
	BackupNamespace string `json:"backupNamespace,omitempty"`
}

// RecoverStatus defines the observed state of Recover
type RecoverStatus struct {
	// the mapping table of old object to new object
	// +kubebuilder:validation:Required
	ObjectMapping map[string]string `json:"objectMapping,omitempty"`

	// state of the recover
	// +kubebuilder:validation:Optional
	State string `json:"state,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Mapping",type=string,JSONPath=`.status.objectMapping`
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.state`

// Recover describes a recover operation of vineyard objects, which is used to recover
// a specific backup operation.
type Recover struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   RecoverSpec   `json:"spec,omitempty"`
	Status RecoverStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// RecoverList contains a list of Recover
type RecoverList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Recover `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Recover{}, &RecoverList{})
}
