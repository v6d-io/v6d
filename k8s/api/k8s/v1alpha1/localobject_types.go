/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// LocalObjectSpec defines the desired state of LocalObject
type LocalObjectSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	ObjectID   string `json:"id"`
	Name       string `json:"name,omitempty"`
	Signature  string `json:"signature"`
	Typename   string `json:"typename,omitempty"`
	InstanceID int    `json:"instance_id"`
	Hostname   string `json:"hostname"`
	Metadata   string `json:"metadata"`
}

// LocalObjectStatus defines the observed state of LocalObject
type LocalObjectStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:categories=all,shortName=lobject
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Id",type=string,JSONPath=`.spec.id`
// +kubebuilder:printcolumn:name="Name",type=string,JSONPath=`.spec.name`
// +kubebuilder:printcolumn:name="Signature",type=string,JSONPath=`.spec.signature`
// +kubebuilder:printcolumn:name="Typename",type=string,JSONPath=`.spec.typename`
// +kubebuilder:printcolumn:name="Instance",type=integer,JSONPath=`.spec.instance_id`
// +kubebuilder:printcolumn:name="Hostname",type=string,JSONPath=`.spec.hostname`
// +genclient

// LocalObject is the Schema for the localobjects API
type LocalObject struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   LocalObjectSpec   `json:"spec,omitempty"`
	Status LocalObjectStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// LocalObjectList contains a list of LocalObject
type LocalObjectList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []LocalObject `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LocalObject{}, &LocalObjectList{})
}
