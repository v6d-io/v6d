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
	// CSIDriverOpts holds all configuration of CSIDriver Spec
	CSIDriverOpts v1alpha1.CSIDriverSpec

	// CSIDriverName is the name of the csi driver cr
	CSIDriverName string

	// VineyardClusters contains all the vineyard clusters
	VineyardClusters []string
)

func ApplyCSIDriverNameOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&CSIDriverName, "name", "", "csidriver-sample",
		"The name of the csi driver cr.")
}
func ApplyCSIDriverSidecarOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&CSIDriverOpts.Sidecar.ProvisionerImage, "provisionerImage", "",
		"registry.k8s.io/sig-storage/csi-provisioner:v3.3.0", "The image of csi provisioner.")
	cmd.Flags().StringVarP(&CSIDriverOpts.Sidecar.AttacherImage, "attacherImage", "",
		"registry.k8s.io/sig-storage/csi-attacher:v4.0.0", "The image of csi attacher.")
	cmd.Flags().StringVarP(&CSIDriverOpts.Sidecar.NodeRegistrarImage, "nodeRegistrarImage", "",
		"registry.k8s.io/sig-storage/csi-node-driver-registrar:v2.6.0", "The image of csi nodeRegistrar.")
	cmd.Flags().StringVarP(&CSIDriverOpts.Sidecar.LivenessProbeImage, "livenessProbeImage", "",
		"registry.k8s.io/sig-storage/livenessprobe:v2.8.0", "The image of livenessProbe.")
	cmd.Flags().StringVarP(&CSIDriverOpts.Sidecar.ImagePullPolicy, "sidecar.imagePullPolicy", "",
		"Always", "The image pull policy of all sidecar containers.")
	cmd.Flags().BoolVarP(&CSIDriverOpts.Sidecar.EnableTopology, "sidecar.enableTopology", "",
		false, "Enable topology for the csi driver.")
}

func ApplyCSIDriverClustersOpts(cmd *cobra.Command) {
	cmd.Flags().StringSliceVarP(&VineyardClusters, "clusters", "",
		[]string{}, "The list of vineyard clusters.")
}

func ApplyCSIDriverOpts(cmd *cobra.Command) {
	ApplyCSIDriverNameOpts(cmd)
	cmd.Flags().StringVarP(&CSIDriverOpts.Image, "image", "i",
		"vineyardcloudnative/vineyard-operator",
		"The image of vineyard csi driver.")
	cmd.Flags().StringVarP(&CSIDriverOpts.ImagePullPolicy, "imagePullPolicy", "",
		"IfNotPresent", "The image pull policy of vineyard csi driver.")
	cmd.Flags().StringVarP(&CSIDriverOpts.StorageClassName, "storageClassName", "s",
		"vineyard-csi", "The name of storage class.")
	cmd.Flags().StringVarP(&CSIDriverOpts.VolumeBindingMode, "volumeBindingMode", "m",
		"WaitForFirstConsumer", "The volume binding mode of the storage class.")
	cmd.Flags().BoolVarP(&CSIDriverOpts.EnableToleration, "enableToleration", "",
		false, "Enable toleration for vineyard csi driver.")
	ApplyCSIDriverSidecarOpts(cmd)
	ApplyCSIDriverClustersOpts(cmd)
}
