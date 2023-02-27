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

	// VineyardSpillPVSpec represent the persistent volume spec of vineyard's spill mechnism
	VineyardSpillPVSpec string

	// VineyardSpillPVCSpec represent the persistent volume claim spec of vineyard's spill mechnism
	VineyardSpillPVCSpec string
)

func ApplyVineyardContainerOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.Image, "vineyard.image",
		"", "vineyardcloudnative/vineyardd:latest", "the image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.ImagePullPolicy,
		"vineyard.imagePullPolicy", "", "IfNotPresent", "the imagePullPolicy of vineyardd")
	cmd.Flags().BoolVarP(&VineyarddOpts.VineyardConfig.SyncCRDs, "vineyard.syncCRDs",
		"", true, "enable metrics of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.Socket, "vineyard.socket",
		"", DefaultVineyardSocket,
		"The directory on host for the IPC socket file. "+
			"The namespace and name will be replaced with your vineyard config")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.Size, "vineyard.size",
		"", "256Mi", "The size of vineyardd. You can use the power-of-two equivalents: "+
			"Ei, Pi, Ti, Gi, Mi, Ki. ")
	cmd.Flags().Int64VarP(&VineyarddOpts.VineyardConfig.StreamThreshold, "vineyard.streamThreshold",
		"", 80, "memory threshold of streams (percentage of total memory)")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.EtcdEndpoint, "vineyard.etcdEndpoint",
		"", "http://etcd-for-vineyard:2379", "The etcd endpoint of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.EtcdPrefix, "vineyard.etcdPrefix",
		"", "/vineyard", "The etcd prefix of vineyardd")
	cmd.Flags().StringSliceVarP(&VineyardContainerEnvs, "envs", "", []string{},
		"The environment variables of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.SpillConfig.Name, "vineyard.spill.config",
		"", "", "If you want to enable the spill mechnism, please set the name of spill config")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.SpillConfig.Path, "vineyard.spill.path",
		"", "", "The path of spill config")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.SpillConfig.SpillLowerRate,
		"vineyard.spill.spillLowerRate",
		"", "0.3", "The low watermark of spilling memorys")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.SpillConfig.SpillUpperRate,
		"vineyard.spill.spillUpperRate",
		"", "0.8", "The high watermark of spilling memorys")
	cmd.Flags().StringVarP(&VineyardSpillPVSpec, "vineyard.spill.pv", "", "",
		"The json string of the persistent volume")
	cmd.Flags().StringVarP(&VineyardSpillPVCSpec, "vineyard.spill.pvc", "", "",
		"The json string of the persistent volume claim")
}

// ApplyServiceOpts represents the option of service
func ApplyServiceOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.Service.Type, "vineyardd.service.type", "", "ClusterIP",
		"the service type of vineyard service")
	cmd.Flags().IntVarP(&VineyarddOpts.Service.Port, "vineyardd.service.port", "", 9600,
		"the service port of vineyard service")
	cmd.Flags().StringVarP(&VineyarddOpts.Service.Selector, "vineyardd.service.selector", "",
		"rpc.vineyardd.v6d.io/rpc=vineyard-rpc", "the service selector of vineyard service")
}

// ApplyVolumeOpts represents the option of pvc volume configuration
func ApplyVolumeOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.Volume.PvcName, "vineyard.volume.pvcname", "",
		"", "Set the pvc name for storing the vineyard objects persistently, ")
	cmd.Flags().StringVarP(&VineyarddOpts.Volume.MountPath, "vineyard.volume.mountPath", "",
		"", "Set the mount path for the pvc")
}

// ApplyMetricContainerOpts represents the option of metric container configuration
func ApplyMetricContainerOpts(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&VineyarddOpts.MetricConfig.Enable, "metric.enable", "",
		false, "enable metrics of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.MetricConfig.Image, "metric.image",
		"", "vineyardcloudnative/vineyard-grok-exporter:latest",
		"the metic image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.MetricConfig.ImagePullPolicy, "metric.imagePullPolicy",
		"", "IfNotPresent", "the imagePullPolicy of the metric image")
}

// ApplyPluginImageOpts represents the option of plugin image configuration
func ApplyPluginImageOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.BackupImage, "plugin.backupImage", "",
		"ghcr.io/v6d-io/v6d/backup-job", "the backup image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.RecoverImage, "plugin.recoverImage", "",
		"ghcr.io/v6d-io/v6d/recover-job", "the recover image of vineyardd")
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
	cmd.Flags().BoolVarP(&VineyarddOpts.CreateServiceAccount, "vineyard.create.serviceAccount",
		"", false, "create service account for vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.ServiceAccountName, "vineyard.serviceAccount.name",
		"", "", "the service account name of vineyardd")
	cmd.Flags().IntVarP(&VineyarddOpts.Etcd.Replicas, "vineyard.etcd.replicas",
		"", 3, "the number of etcd replicas in a vineyard cluster")
	// setup the vineyardd name
	ApplyVineyarddNameOpts(cmd)
	// setup the vineyard container configuration of vineyardd
	ApplyVineyardContainerOpts(cmd)
	// setup the metric container configuration of vineyardd
	ApplyMetricContainerOpts(cmd)
	// setup the vineyard service configuration of vineyardd
	ApplyServiceOpts(cmd)
	// setup the vineyard volumes if needed
	ApplyVolumeOpts(cmd)
	// setup the plugin images in a vineyard workflow
	ApplyPluginImageOpts(cmd)
}
