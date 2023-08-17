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

// OperationSpec defines the desired state of Operation
type OperationSpec struct {
	// the name of vineyard pluggable drivers, including assembly and repartition.
	// +kubebuilder:validation:Required
	Name string `json:"name,omitempty"`
	// the type of object, including local and distributed.
	// +kubebuilder:validation:Required
	Type string `json:"type,omitempty"`
	// the required job's name of the operation
	// +kubebuilder:validation:Required
	Require string `json:"require,omitempty"`
	// the target job's name of the operation
	// +kubebuilder:validation:Required
	Target string `json:"target,omitempty"`
	// TimeoutSeconds is the timeout of the operation.
	// +kubebuilder:validation:Required
	TimeoutSeconds int64 `json:"timeoutSeconds,omitempty"`
}

// OperationStatus defines the observed state of Operation
type OperationStatus struct {
	// the state of operation.
	// +kubebuilder:validation:Optional
	State string `json:"state,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Operation",type=string,JSONPath=`.spec.name`
// +kubebuilder:printcolumn:name="Type",type=string,JSONPath=`.spec.type`
// +kubebuilder:printcolumn:name="State",type=string,JSONPath=`.status.state`

// Operation describes an operation between workloads, such as assembly and repartition.
//
// As for the `assembly` operation, there are several kinds of computing engines, some
// may not support the stream data, so we need to insert an `assembly` operation to
// assemble the stream data into a batch data, so that the next computing engines can
// process the data.
//
// As for the `repartition` operation, the vineyard has integrated with the distributed
// computing engines, such as Dask. If you want to repartition the data to adapt the dask
// workers, then the `repartition` operation is essential for such scenario.
type Operation struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   OperationSpec   `json:"spec,omitempty"`
	Status OperationStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// OperationList contains a list of Operation
type OperationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Operation `json:"items"`
}

const (
	// OperationRunning is the running state of the job
	OperationRunning = "running"
	// OperationSucceeded is the succeeded state of the job
	OperationSucceeded = "succeeded"
	// OperationFailed is the failed state of the job
	OperationFailed = "failed"
)

func init() {
	SchemeBuilder.Register(&Operation{}, &OperationList{})
}
