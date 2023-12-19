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
	"bytes"
	"text/template"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

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
	// +kubebuilder:default:={}
	PersistentVolumeSpec corev1.PersistentVolumeSpec `json:"persistentVolumeSpec,omitempty"`

	// the PersistentVolumeClaimSpec of the spill file
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	PersistentVolumeClaimSpec corev1.PersistentVolumeClaimSpec `json:"persistentVolumeClaimSpec,omitempty"`
}

// ServiceConfig holds all service configuration about vineyardd
type ServiceConfig struct {
	// service type
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="ClusterIP"
	Type string `json:"type,omitempty"`

	// service port
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=9600
	Port int `json:"port,omitempty"`
}

// MetricConfig holds the configuration about metric container
type MetricConfig struct {
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

// VineyardConfig holds all configuration about vineyard container
type VineyardConfig struct {
	// represent the vineyardd's image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="vineyardcloudnative/vineyardd:latest"
	Image string `json:"image,omitempty"`

	// the policy about pulling image
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// synchronize CRDs when persisting objects
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=true
	SyncCRDs bool `json:"syncCRDs,omitempty"`

	// The directory on host for the IPC socket file. The UNIX-domain
	// socket will be placed as `${Socket}/vineyard.sock`.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}"
	Socket string `json:"socket,omitempty"`

	// shared memory size for vineyardd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=""
	Size string `json:"size,omitempty"`

	// reserve the shared memory for vineyardd
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=false
	ReserveMemory bool `json:"reserveMemory,omitempty"`

	// memory threshold of streams (percentage of total memory)
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=80
	StreamThreshold int64 `json:"streamThreshold,omitempty"`

	// the configuration of spilling
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Spill SpillConfig `json:"spill,omitempty"`

	// vineyard environment configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Env []corev1.EnvVar `json:"env,omitempty"`

	// the memory resources of vineyard container
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Memory string `json:"memory,omitempty"`

	// the cpu resources of vineyard container
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	CPU string `json:"cpu,omitempty"`
}

// PluginImageConfig holds all image configuration about pluggable drivers(backup, recover,
// local assembly, distributed assembly, repartition)
type PluginImageConfig struct {
	// the image of backup operation
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="ghcr.io/v6d-io/v6d/backup-job"
	BackupImage string `json:"backupImage,omitempty"`

	// the image of recover operation
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="ghcr.io/v6d-io/v6d/recover-job"
	RecoverImage string `json:"recoverImage,omitempty"`

	// the image of dask repartition operation
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="ghcr.io/v6d-io/v6d/dask-repartition"
	DaskRepartitionImage string `json:"daskRepartitionImage,omitempty"`

	// the image of local assembly operation
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="ghcr.io/v6d-io/v6d/local-assembly"
	LocalAssemblyImage string `json:"localAssemblyImage,omitempty"`

	// the image of distributed assembly operation
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:="ghcr.io/v6d-io/v6d/distributed-assembly"
	DistributedAssemblyImage string `json:"distributedAssemblyImage,omitempty"`
}

// VineyarddSpec holds all configuration about vineyardd
type VineyarddSpec struct {
	// Replicas is the number of vineyardd pods to deploy
	// +kubebuilder:validation:Required
	// +kubebuilder:default:=3
	Replicas int `json:"replicas,omitempty"`

	// EtcdReplicas describe the etcd replicas
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:=1
	EtcdReplicas int `json:"etcdReplicas,omitempty"`

	// vineyardd's service
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={type: "ClusterIP", port: 9600}
	Service ServiceConfig `json:"service,omitempty"`

	// vineyard container configuration
	// +kubebuilder:validation:Optional
	//nolint: lll
	// +kubebuilder:default:={image: "vineyardcloudnative/vineyardd:latest", imagePullPolicy: "IfNotPresent", syncCRDs: true, socket: "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}", size: "", streamThreshold: 80}
	Vineyard VineyardConfig `json:"vineyard,omitempty"`

	// operation container configuration
	// +kubebuilder:validation:Optional
	//nolint: lll
	// +kubebuilder:default={backupImage: "ghcr.io/v6d-io/v6d/backup-job", recoverImage: "ghcr.io/v6d-io/v6d/recover-job", daskRepartitionImage: "ghcr.io/v6d-io/v6d/dask-repartition", localAssemblyImage: "ghcr.io/v6d-io/v6d/local-assembly", distributedAssemblyImage: "ghcr.io/v6d-io/v6d/distributed-assembly"}
	PluginImage PluginImageConfig `json:"pluginImage,omitempty"`

	// metric container configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={enable: false, image: "vineyardcloudnative/vineyard-grok-exporter:latest", imagePullPolicy: "IfNotPresent"}
	Metric MetricConfig `json:"metric,omitempty"`

	// Volume configuration
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={pvcName: "", mountPath: ""}
	Volume VolumeConfig `json:"volume,omitempty"`

	// SecurityContext holds the security context settings for the vineyardd container.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	SecurityContext corev1.SecurityContext `json:"securityContext,omitempty"`

	// Volumes is the list of Kubernetes volumes that can be mounted by the vineyard deployment.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// VolumeMounts specifies the volumes listed in ".spec.volumes" to mount into the vineyard deployment.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default:={}
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`
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

// Vineyardd is used to deploy a vineyard cluster on kubernetes, which can simplify the
// configurations of the vineyard binary, the external etcd cluster and the
// vineyard [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/).
// As vineyard is bound to a specific socket on the hostpath by default, the vineyard pod cannot be
// deployed on the same node. Before deploying vineyardd, you should know how many nodes
// are available for vineyard pod to deploy on and make sure the vineyardd pod number is
// less than the number of available nodes.
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
	if vineyardd.Spec.Vineyard.Socket == "" {
		return
	}
	if tpl, err := template.New("vineyardd").Parse(vineyardd.Spec.Vineyard.Socket); err == nil {
		var buf bytes.Buffer
		if err := tpl.Execute(&buf, vineyardd); err == nil {
			vineyardd.Spec.Vineyard.Socket = buf.String()
		}
	}
}

func init() {
	SchemeBuilder.Register(&Vineyardd{}, &VineyarddList{})
}
