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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// VineyardJobSpec defines the desired state of VineyardJob
type VineyardJobSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Replicas int                    `json:"replicas,omitempty"`
	Template corev1.PodTemplateSpec `json:"template,omitempty"`
}

// VineyardJobStatus defines the observed state of VineyardJob
type VineyardJobStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Replicas int      `json:"replicas,omitempty"`
	Ready    int      `json:"ready,omitempty"`
	Hosts    []string `json:"hosts,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:categories=all,shortName=vineyardjob
// +kubebuilder:printcolumn:name="replicas",type=integer,JSONPath=`.status.replicas`
// +kubebuilder:printcolumn:name="ready",type=integer,JSONPath=`.status.ready`
// +kubebuilder:printcolumn:name="hosts",type=string,JSONPath=`.status.hosts`
// +genclient

// VineyardJob is the Schema for the vineyardjobs API
type VineyardJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VineyardJobSpec   `json:"spec,omitempty"`
	Status VineyardJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// VineyardJobList contains a list of VineyardJob
type VineyardJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VineyardJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&VineyardJob{}, &VineyardJobList{})
}
