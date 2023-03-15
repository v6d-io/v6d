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
)

var (
	// DefaultVineyardSocket is the default vineyard socket path
	DefaultVineyardSocket = "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}"

	// VineyarddName is the name of vineyardd
	VineyarddName string

	// VineyarddOpts holds all configuration of vineyardd Spec
	VineyarddOpts v1alpha1.VineyarddSpec

	// VineyardContainerEnvs holds all the environment variables for the vineyardd container
	VineyardContainerEnvs []string

	// VineyardSpillPVAndPVC is PersistentVolume data and PersistentVolumeClaim data of vineyardd's spill mechanism
	VineyardSpillPVandPVC string

	// VineyardFile is the path of vineyardd file
	VineyarddFile string
)

// ApplyVineyardContainerOpts applies the vineyard container options
func ApplyVineyardContainerOpts(c *v1alpha1.VineyardContainerConfig,
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
		"", "256Mi",
		"The size of vineyardd. You can use the power-of-two equivalents: "+
			"Ei, Pi, Ti, Gi, Mi, Ki. ")
	cmd.Flags().Int64VarP(&c.StreamThreshold, prefix+".streamThreshold",
		"", 80, "memory threshold of streams (percentage of total memory)")
	cmd.Flags().StringVarP(&c.EtcdEndpoint, prefix+".etcdEndpoint",
		"", "http://etcd-for-vineyard:2379", "The etcd endpoint of vineyardd")
	cmd.Flags().StringVarP(&c.EtcdPrefix, prefix+".etcdPrefix",
		"", "/vineyard", "The etcd prefix of vineyardd")
	cmd.Flags().StringSliceVarP(&VineyardContainerEnvs, prefix+".envs", "", []string{},
		"The environment variables of vineyardd")
	cmd.Flags().StringVarP(&c.SpillConfig.Name, prefix+".spill.config",
		"", "",
		"If you want to enable the spill mechanism, please set the name of spill config")
	cmd.Flags().StringVarP(&c.SpillConfig.Path, prefix+".spill.path",
		"", "", "The path of spill config")
	cmd.Flags().StringVarP(&c.SpillConfig.SpillLowerRate,
		prefix+".spill.spillLowerRate",
		"", "0.3", "The low watermark of spilling memory")
	cmd.Flags().StringVarP(&c.SpillConfig.SpillUpperRate,
		prefix+".spill.spillUpperRate",
		"", "0.8", "The high watermark of spilling memory")
	cmd.Flags().StringVarP(&VineyardSpillPVandPVC, prefix+".spill.pv-pvc-spec", "", "",
		"the json string of the persistent volume and persistent volume claim")
}

// ApplyServiceOpts represents the option of service
func ApplyServiceOpts(s *v1alpha1.ServiceConfig, prefix string, cmd *cobra.Command) {
	cmd.Flags().StringVarP(&s.Type, prefix+".service.type", "", "ClusterIP",
		"the service type of vineyard service")
	cmd.Flags().IntVarP(&s.Port, prefix+".service.port", "", 9600,
		"the service port of vineyard service")
	cmd.Flags().StringVarP(&s.Selector, prefix+".service.selector", "",
		"rpc.vineyardd.v6d.io/rpc=vineyard-rpc",
		"the service selector of vineyard service")
}

// ApplyVolumeOpts represents the option of pvc volume configuration
func ApplyVolumeOpts(v *v1alpha1.VolumeConfig, prefix string, cmd *cobra.Command) {
	cmd.Flags().StringVarP(&v.PvcName, prefix+".volume.pvcname", "",
		"", "Set the pvc name for storing the vineyard objects persistently, ")
	cmd.Flags().StringVarP(&v.MountPath, prefix+".volume.mountPath", "",
		"", "Set the mount path for the pvc")
}

// ApplyMetricContainerOpts represents the option of metric container configuration
func ApplyMetricContainerOpts(m *v1alpha1.MetricContainerConfig,
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
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.BackupImage,
		"plugin.backupImage", "", "ghcr.io/v6d-io/v6d/backup-job",
		"the backup image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.RecoverImage,
		"plugin.recoverImage", "", "ghcr.io/v6d-io/v6d/recover-job",
		"the recover image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.DaskRepartitionImage,
		"plugin.daskRepartitionImage", "", "ghcr.io/v6d-io/v6d/dask-repartition",
		"the dask repartition image of vineyardd workflow")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.LocalAssemblyImage,
		"plugin.localAssemblyImage", "", "ghcr.io/v6d-io/v6d/local-assembly",
		"the local assembly image of vineyardd workflow")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.DistributedAssemblyImage,
		"plugin.distributedAssemblyImage", "", "ghcr.io/v6d-io/v6d/distributed-assembly",
		"the distributed image of vineyard workflow")
}

// ApplyVineyarddNameOpts represents the option of vineyardd name
func ApplyVineyarddNameOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddName, "name", "", "vineyardd-sample",
		"the name of vineyardd")
}

// ApplyVineyarddOpts represents the option of vineyardd configuration
func ApplyVineyarddOpts(cmd *cobra.Command) {
	// setup the vineyardd configuration
	cmd.Flags().IntVarP(&VineyarddOpts.Replicas, "vineyard.replicas", "", 3,
		"the number of vineyardd replicas")
	cmd.Flags().BoolVarP(&VineyarddOpts.CreateServiceAccount,
		"vineyard.create.serviceAccount", "", false,
		"create service account for vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.ServiceAccountName,
		"vineyard.serviceAccount.name",
		"", "", "the service account name of vineyardd")
	cmd.Flags().IntVarP(&VineyarddOpts.Etcd.Replicas, "vineyard.etcd.replicas",
		"", 3, "the number of etcd replicas in a vineyard cluster")
	cmd.Flags().StringVarP(&VineyarddFile, "file", "f", "", "the path of vineyardd")
	// setup the vineyardd name
	ApplyVineyarddNameOpts(cmd)
	// setup the vineyard container configuration of vineyardd
	ApplyVineyardContainerOpts(&VineyarddOpts.VineyardConfig, "vineyardd", cmd)
	// setup the metric container configuration of vineyardd
	ApplyMetricContainerOpts(&VineyarddOpts.MetricConfig, "vineyardd", cmd)
	// setup the vineyard service configuration of vineyardd
	ApplyServiceOpts(&VineyarddOpts.Service, "vineyardd", cmd)
	// setup the vineyard volumes if needed
	ApplyVolumeOpts(&VineyarddOpts.Volume, "vineyardd", cmd)
	// setup the plugin images in a vineyard workflow
	ApplyPluginImageOpts(cmd)
}
