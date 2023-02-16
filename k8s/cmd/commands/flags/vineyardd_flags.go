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

// VineyarddName is the name of vineyardd
var VineyarddName string

// VineyarddOpts holds all configuration of vineyardd Spec
var VineyarddOpts v1alpha1.VineyarddSpec

// VineyardContainerEnvs holds all the environment variables for the vineyardd container
var VineyardContainerEnvs []string

// VineyardSpillPVSpec represent the persistent volume spec of vineyard's spill mechnism
var VineyardSpillPVSpec string

// VineyardSpillPVCSpec represent the persistent volume claim spec of vineyard's spill mechnism
var VineyardSpillPVCSpec string

func NewVineyardContainerOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.Image, "vineyard.image",
		"", "vineyardcloudnative/vineyardd:latest", "the image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.ImagePullPolicy,
		"vineyard.imagePullPolicy", "", "IfNotPresent", "the imagePullPolicy of vineyardd")
	cmd.Flags().BoolVarP(&VineyarddOpts.VineyardConfig.SyncCRDs, "vineyard.syncCRDs",
		"", true, "enable metrics of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.Socket, "vineyard.socket",
		"", "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}",
		"The directory on host for the IPC socket file. "+
			"The namespace and name will be replaced with your vineyard config")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.Size, "vineyard.size",
		"", "256Mi", "The size of vineyardd. You can use the power-of-two equivalents: "+
			"Ei, Pi, Ti, Gi, Mi, Ki. ")
	cmd.Flags().Int64VarP(&VineyarddOpts.VineyardConfig.StreamThreshold, "vineyard.streamThreshold",
		"", 80, "memory threshold of streams (percentage of total memory)")
	cmd.Flags().StringVarP(&VineyarddOpts.VineyardConfig.EtcdEndpoint, "vineyard.etcdEndpoint",
		"", "", "The etcd endpoint of vineyardd")
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

// NewServiceOpts represents the option of service
func NewServiceOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.Service.Type, "vineyardd.service.type", "", "ClusterIP",
		"the service type of vineyard service")
	cmd.Flags().IntVarP(&VineyarddOpts.Service.Port, "vineyardd.service.port", "", 9600,
		"the service port of vineyard service")
	cmd.Flags().StringVarP(&VineyarddOpts.Service.Selector, "vineyardd.service.selector", "",
		"rpc.vineyardd.v6d.io/rpc=vineyard-rpc", "the service selector of vineyard service")
}

// NewVolumeOpts represents the option of pvc volume configuration
func NewVolumeOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.Volume.PvcName, "vineyard.volume.pvcname", "",
		"", "Set the pvc name for storing the vineyard objects persistently, ")
	cmd.Flags().StringVarP(&VineyarddOpts.Volume.MountPath, "vineyard.volume.mountPath", "",
		"", "Set the mount path for the pvc")
}

// NewMetricContainerOpts represents the option of metric container configuration
func NewMetricContainerOpts(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&VineyarddOpts.MetricConfig.Enable, "metric.enable", "",
		false, "enable metrics of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.MetricConfig.Image, "metric.image",
		"", "vineyardcloudnative/vineyard-grok-exporter:latest",
		"the metic image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.MetricConfig.ImagePullPolicy, "metric.imagePullPolicy",
		"", "IfNotPresent", "the imagePullPolicy of the metric image")
}

// NewPluginImageOpts represents the option of plugin image configuration
func NewPluginImageOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.BackupImage, "plugin.backupimage", "",
		"ghcr.io/v6d-io/v6d/backup-job", "the backup image of vineyardd")
	cmd.Flags().StringVarP(&VineyarddOpts.PluginConfig.RecoverImage, "plugin.recoverimage", "",
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

// NewVineyarddNameOpts represents the option of vineyardd name
func NewVineyarddNameOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&VineyarddName, "name", "", "vineyardd-sample",
		"the name of vineyardd")
}

// NewVineyarddOpts represents the option of vineyardd configuration
func NewVineyarddOpts(cmd *cobra.Command) {
	// setup the vineyardd configuration
	cmd.Flags().IntVarP(&VineyarddOpts.Replicas, "vineyard.replicas", "", 3,
		"the number of vineyardd replicas")
	cmd.Flags().IntVarP(&VineyarddOpts.Etcd.Replicas, "vineyard.etcd.replicas",
		"", 3, "the number of etcd replicas in a vineyard cluster")
	// setup the vineyardd name
	NewVineyarddNameOpts(cmd)
	// setup the vineyard container configuration of vineyardd
	NewVineyardContainerOpts(cmd)
	// setup the metric container configuration of vineyardd
	NewMetricContainerOpts(cmd)
	// setup the vineyard service configuration of vineyardd
	NewServiceOpts(cmd)
	// setup the vineyard volumes if needed
	NewVolumeOpts(cmd)
	// setup the plugin images in a vineyard workflow
	NewPluginImageOpts(cmd)
}
