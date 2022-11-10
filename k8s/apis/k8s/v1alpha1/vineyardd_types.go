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
	"bytes"
	"text/template"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MetricConfig holds all metrics configuration
type MetricConfig struct {
	// represent the metric's image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default="vineyardcloudnative/vineyard-grok-exporter:latest"
	Image string `json:"image,omitempty"`
	// the policy about pulling image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`
}

// VineyarddConfig holds all command configuration about vineyardd
// SpillConfig holds all configuration about spilling
type SpillConfig struct {
	// the name of the spill config
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	Name string `json:"name,omitempty"`
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
	// the PersistentVolumeSpec of the spilling PV
	// +kubebuilder:validation:Optional
	PersistentVolumeSpec corev1.PersistentVolumeSpec `json:"persistentVolumeSpec,omitempty"`
	// the PersistentVolumeClaimSpec of the spill file
	// +kubebuilder:validation:Optional
	PersistentVolumeClaimSpec corev1.PersistentVolumeClaimSpec `json:"persistentVolumeClaimSpec,omitempty"`
}

// ServiceConfig holds all service configuration about vineyardd
type ServiceConfig struct {
	// service type
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=ClusterIP
	Type string `json:"type,omitempty"`
	// service port
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=9600
	Port int `json:"port"`
}

// Etcd holds all configuration about Etcd
type EtcdConfig struct {
	// Etcd replicas
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=3
	Replicas int `json:"replicas"`
}

// VineyarddSpec holds all configuration about vineyardd
type VineyarddSpec struct {
	// represent the vineyardd's image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="vineyardcloudnative/vineyardd:latest"
	Image string `json:"image,omitempty"`
	// the policy about pulling image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`
	// the replicas of vineyardd
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=3
	Replicas int `json:"replicas"`

	// synchronize CRDs when persisting objects
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=true
	SyncCRDs bool `json:"syncCRDs"`

	// The directory on host for the IPC socket file. The UNIX-domain
	// socket will be placed as `${Socket}/vineyard.sock`.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="/var/run/vineyard-kubernetes/{{ .Namespace }}/{{ .Name }}"
	Socket string `json:"socket"`
	// vineyardd's service
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={type: "ClusterIP", port: 9600}
	Service ServiceConfig `json:"service,omitempty"`
	// shared memory size for vineyardd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="256Mi"
	SharedMemorySize string `json:"sharedMemorySize"`
	// memory threshold of streams (percentage of total memory)
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=80
	StreamThreshold int64 `json:"streamThreshold"`

	// path of etcd executable
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="etcd"
	EtcdCmd string `json:"etcdCmd"`
	// endpoint of etcd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="http://etcd-for-vineyard:2379"
	EtcdEndpoint string `json:"etcdEndpoint"`
	// path prefix in etcd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="/vineyard"
	EtcdPrefix string `json:"etcdPrefix"`
	// Etcd describe the etcd replicas
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={replicas: 3}
	Etcd EtcdConfig `json:"etcd,omitempty"`

	// the configuration of spilling
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	SpillConfig SpillConfig `json:"spillConfig,omitempty"`

	// enable the metrics in vineyardd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=false
	EnableMetrics bool `json:"enableMetrics"`
	// metric configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={image: "vineyardcloudnative/vineyard-grok-exporter:latest", imagePullPolicy: "IfNotPresent"}
	Metric MetricConfig `json:"metric,omitempty"`

	// environment configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// VineyarddStatus defines the observed state of Vineyardd
type VineyarddStatus struct {
	// Total replicas of current running vineyardd.
	ReadyReplicas int32 `json:"current,omitempty"`
	// Represents the vineyardd deployment's current state.
	Conditions []appsv1.DeploymentCondition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Current",type=string,JSONPath=`.status.current`
// +kubebuilder:printcolumn:name="Desired",type=string,JSONPath=`.spec.replicas`
// +genclient

// Vineyardd is the Schema for the vineyardd API
type Vineyardd struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VineyarddSpec   `json:"spec,omitempty"`
	Status VineyarddStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// VineyarddList contains a list of Vineyardd
type VineyarddList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Vineyardd `json:"items"`
}

func PreprocessVineyarddSocket(vineyardd *Vineyardd) {
	if vineyardd.Spec.Socket == "" {
		return
	}
	if tpl, err := template.New("vineyardd").Parse(vineyardd.Spec.Socket); err == nil {
		var buf bytes.Buffer
		if err := tpl.Execute(&buf, vineyardd); err == nil {
			vineyardd.Spec.Socket = buf.String()
		}
	}
}

func init() {
	SchemeBuilder.Register(&Vineyardd{}, &VineyarddList{})
}
