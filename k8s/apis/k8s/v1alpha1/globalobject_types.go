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

// GlobalObjectSpec defines the desired state of GlobalObject
type GlobalObjectSpec struct {
	// +kubebuilder:validation:Required
	ObjectID string `json:"id"`
	// +kubebuilder:validation:Required
	Name string `json:"name,omitempty"`
	// +kubebuilder:validation:Required
	Signature string `json:"signature"`
	// +kubebuilder:validation:Required
	Typename string `json:"typename,omitempty"`
	// +kubebuilder:validation:Optional
	Members []string `json:"members"`
	// +kubebuilder:validation:Optional
	Metadata string `json:"metadata"`
}

// GlobalObjectStatus defines the observed state of GlobalObject
type GlobalObjectStatus struct {
	// The state represents the current state of the global object.
	// +kubebuilder:validation:Optional
	State string `json:"state"`
	// The time when the global object is created.
	// +kubebuilder:validation:Optional
	CreationTime metav1.Time `json:"createdTime"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:categories=all,shortName=gobject
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Id",type=string,JSONPath=`.spec.id`
// +kubebuilder:printcolumn:name="Name",type=string,JSONPath=`.spec.name`
// +kubebuilder:printcolumn:name="Signature",type=string,JSONPath=`.spec.signature`
// +kubebuilder:printcolumn:name="Typename",type=string,JSONPath=`.spec.typename`
// +genclient

// GlobalObject describes a global object in vineyard, whose metadata
// will be stored in etcd.
type GlobalObject struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   GlobalObjectSpec   `json:"spec,omitempty"`
	Status GlobalObjectStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GlobalObjectList contains a list of GlobalObject
type GlobalObjectList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GlobalObject `json:"items"`
}

func init() {
	SchemeBuilder.Register(&GlobalObject{}, &GlobalObjectList{})
}
