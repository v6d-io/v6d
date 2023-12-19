/*
* Copyright 2020-2023 Alibaba Group Holding Limited.

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
package flags

import (
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

type VolumeConfig struct {
	Volumes      []corev1.Volume      `yaml:"volumes"`
	VolumeMounts []corev1.VolumeMount `yaml:"volumeMounts"`
}

var (
	// DefaultVineyardSocket is the default vineyard socket path
	DefaultVineyardSocket = "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}"

	// VineyarddName is the name of vineyardd
	VineyarddName string

	// VineyardSecurityContext is the json string of security context of vineyardd
	VineyardSecurityContext string

	// VineyarddOpts holds all configuration of vineyardd Spec
	VineyarddOpts v1alpha1.VineyarddSpec

	// VineyardContainerEnvs holds all the environment variables for the vineyardd container
	VineyardContainerEnvs []string

	// VineyardSpillPVAndPVC is PersistentVolume data and PersistentVolumeClaim data of vineyardd's spill mechanism
	VineyardSpillPVandPVC string

	// VineyardFile is the path of vineyardd file
	VineyarddFile string

	// The following variables are used for the vineyard client
	// VineyardIPCSocket is the path of vineyardd IPC socket
	VineyardIPCSocket string

	// VineyardRPCSocket is the path of vineyardd RPC socket
	VineyardRPCSocket string

	// VineyardVolume is the json string of vineyardd volume
	VineyardVolume string

	// VineyardVolumeMount is the json string of vineyardd volume mount
	VineyardVolumeMount string

	// NamespacedVineyardDeployment is the namespaced name of vineyard deployment
	NamespacedVineyardDeployment string
)

// ApplyVineyardContainerOpts applies the vineyard container options
func ApplyVineyardContainerOpts(c *v1alpha1.VineyardConfig,
	prefix string, cmd *cobra.Command,
) {
	cmd.Flags().StringVarP(&c.Image, prefix+".image",
		"", "vineyardcloudnative/vineyardd:latest", "the image of vineyardd")
	cmd.Flags().StringVarP(&c.ImagePullPolicy,
		prefix+".imagePullPolicy", "", "IfNotPresent",
		"the imagePullPolicy of vineyardd")
	cmd.Flags().BoolVarP(&c.SyncCRDs, prefix+".syncCRDs",
		"", true, "enable metrics of vineyardd")
	cmd.Flags().StringVarP(&c.Socket, prefix+".socket",
		"", DefaultVineyardSocket,
		"The directory on host for the IPC socket file. "+
			"The namespace and name will be replaced with your vineyard config")
	cmd.Flags().StringVarP(&c.Size, prefix+".size",
		"", "",
		"The size of vineyardd. You can use the power-of-two equivalents: "+
			"Ei, Pi, Ti, Gi, Mi, Ki. Defaults \"\", means not limited")
	cmd.Flags().BoolVarP(&c.ReserveMemory, prefix+".reserve_memory",
		"", false,
		"Reserving enough physical memory pages for vineyardd")
	cmd.Flags().Int64VarP(&c.StreamThreshold, prefix+".streamThreshold",
		"", 80, "memory threshold of streams (percentage of total memory)")
	cmd.Flags().StringSliceVarP(&VineyardContainerEnvs, prefix+".envs", "", []string{},
		"The environment variables of vineyardd")
	cmd.Flags().StringVarP(&c.Spill.Name, prefix+".spill.config",
		"", "",
		"If you want to enable the spill mechanism, please set the name of spill config")
	cmd.Flags().StringVarP(&c.Spill.Path, prefix+".spill.path",
		"", "", "The path of spill config")
	cmd.Flags().StringVarP(&c.Spill.SpillLowerRate,
		prefix+".spill.spillLowerRate",
		"", "0.3", "The low watermark of spilling memory")
	cmd.Flags().StringVarP(&c.Spill.SpillUpperRate,
		prefix+".spill.spillUpperRate",
		"", "0.8", "The high watermark of spilling memory")
	cmd.Flags().StringVarP(&VineyardSpillPVandPVC, prefix+".spill.pv-pvc-spec", "", "",
		"the json string of the persistent volume and persistent volume claim")
	cmd.Flags().StringVarP(&c.Memory, prefix+".memory", "", "",
		"the memory requests and limits of vineyard container")
	cmd.Flags().StringVarP(&c.CPU, prefix+".cpu", "", "",
		"the cpu requests and limits of vineyard container")
}

// ApplyServiceOpts represents the option of service
func ApplyServiceOpts(s *v1alpha1.ServiceConfig, prefix string, cmd *cobra.Command) {
	cmd.Flags().StringVarP(&s.Type, prefix+".service.type", "", "ClusterIP",
		"the service type of vineyard service")
	cmd.Flags().IntVarP(&s.Port, prefix+".service.port", "", 9600,
		"the service port of vineyard service")
}

// ApplyVolumeOpts represents the option of pvc volume configuration
func ApplyVolumeOpts(v *v1alpha1.VolumeConfig, prefix string, cmd *cobra.Command) {
	cmd.Flags().StringVarP(&v.PvcName, prefix+".volume.pvcname", "",
		"", "Set the pvc name for storing the vineyard objects persistently")
	cmd.Flags().StringVarP(&v.MountPath, prefix+".volume.mountPath", "",
		"", "Set the mount path for the pvc")
}

// ApplyMetricContainerOpts represents the option of metric container configuration
func ApplyMetricContainerOpts(m *v1alpha1.MetricConfig,
	prefix string, cmd *cobra.Command,
) {
	cmd.Flags().BoolVarP(&m.Enable, prefix+".metric.enable", "",
		false, "enable metrics of vineyardd")
	cmd.Flags().StringVarP(&m.Image, prefix+".metric.image",
		"", "vineyardcloudnative/vineyard-grok-exporter:latest",
		"the metic image of vineyardd")
	cmd.Flags().StringVarP(&m.ImagePullPolicy, prefix+".metric.imagePullPolicy",
		"", "IfNotPresent", "the imagePullPolicy of the metric image")
}

// ApplyPluginImageOpts represents the option of plugin image configuration
func ApplyPluginImageOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.PluginImage.BackupImage,
		"pluginImage.backupImage", "", "ghcr.io/v6d-io/v6d/backup-job",
		"the backup image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginImage.RecoverImage,
		"pluginImage.recoverImage", "", "ghcr.io/v6d-io/v6d/recover-job",
		"the recover image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginImage.DaskRepartitionImage,
		"pluginImage.daskRepartitionImage", "", "ghcr.io/v6d-io/v6d/dask-repartition",
		"the dask repartition image of vineyardd workflow")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginImage.LocalAssemblyImage,
		"pluginImage.localAssemblyImage", "", "ghcr.io/v6d-io/v6d/local-assembly",
		"the local assembly image of vineyardd workflow")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginImage.DistributedAssemblyImage,
		"pluginImage.distributedAssemblyImage", "", "ghcr.io/v6d-io/v6d/distributed-assembly",
		"the distributed image of vineyard workflow")
}

// ApplyVineyarddNameOpts represents the option of vineyardd name
func ApplyVineyarddNameOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddName, "name", "", "vineyardd-sample",
		"the name of vineyardd")
}

// ApplyVineyardVolumeAndVolumeMountOpts represents the option of vineyardd volume and volume mount
func ApplyVineyardVolumeAndVolumeMountOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyardVolume, "volume", "", "",
		"the json string of vineyardd volume")
	cmd.Flags().StringVarP(&VineyardVolumeMount, "volumeMount", "", "",
		"the json string of vineyardd volume mount")
}

// ApplyVineyarddSecurityContextOpts represents the option of vineyard security context
func ApplyVineyarddPrivilegedOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyardSecurityContext, "securityContext", "", "",
		"the json string of security context of vineyardd")
}

// ApplyVineyarddOpts represents the option of vineyardd configuration
func ApplyVineyarddOpts(cmd *cobra.Command) {
	// setup the vineyardd configuration
	cmd.Flags().IntVarP(&VineyarddOpts.Replicas, "replicas", "", 3,
		"the number of vineyardd replicas")
	cmd.Flags().IntVarP(&VineyarddOpts.EtcdReplicas, "etcd.replicas",
		"", 1, "the number of etcd replicas in a vineyard cluster")
	cmd.Flags().StringVarP(&VineyarddFile, "file", "f", "", "the path of vineyardd")
	// setup the vineyardd name
	ApplyVineyarddNameOpts(cmd)
	// setup the vineyard container configuration of vineyardd
	ApplyVineyardContainerOpts(&VineyarddOpts.Vineyard, "vineyardd", cmd)
	// setup the metric container configuration of vineyardd
	ApplyMetricContainerOpts(&VineyarddOpts.Metric, "vineyardd", cmd)
	// setup the vineyard service configuration of vineyardd
	ApplyServiceOpts(&VineyarddOpts.Service, "vineyardd", cmd)
	// setup the vineyard socket volumes if needed
	ApplyVolumeOpts(&VineyarddOpts.Volume, "vineyardd", cmd)
	// setup the plugin images in a vineyard workflow
	ApplyPluginImageOpts(cmd)
	// setup the privileged of vineyard container
	ApplyVineyarddPrivilegedOpts(cmd)
	// setup the vineyardd volume and volume mount
	ApplyVineyardVolumeAndVolumeMountOpts(cmd)
}
