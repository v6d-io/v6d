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
	// SidecarName is the name of sidecar
	// it is also the label selector value of sidecar
	SidecarName string

	// WorkloadYaml is the yaml of workload
	WorkloadYaml string

	// WorkloadResource is the resource of workload
	WorkloadResource string

	// OutputFormat is the output format of the command
	OutputFormat string

	// ApplyResources means whether to apply the resources
	// including the etcd cluster and the rpc service
	ApplyResources bool

	// OwnerReference is the owner reference of the workload
	OwnerReference string

	// SidecarOpts holds all configuration of sidecar Spec
	SidecarOpts v1alpha1.SidecarSpec
)

func ApplySidecarOpts(cmd *cobra.Command) {
	// setup the vineyard container configuration of vineyard sidecar
	ApplyVineyardContainerOpts(&SidecarOpts.Vineyard, "sidecar", cmd)
	// setup the metric container configuration of vineyard sidecar
	ApplyMetricContainerOpts(&SidecarOpts.Metric, "sidecar", cmd)
	// setup the vineyard service configuration of vineyard sidecar
	ApplyServiceOpts(&SidecarOpts.Service, "sidecar", cmd)
	// setup the vineyard volumes if needed
	ApplyVolumeOpts(&SidecarOpts.Volume, "sidecar", cmd)
	cmd.Flags().StringVarP(&SidecarName, "name", "", "vineyard-sidecar",
		"The name of sidecar")
	cmd.Flags().IntVarP(&SidecarOpts.Replicas, "etcd-replicas", "", 1,
		"The number of etcd replicas")
	cmd.Flags().StringVarP(&WorkloadYaml, "file", "f", "", "The yaml of workload")
	cmd.Flags().StringVarP(&WorkloadResource, "resource", "", "", "The resource of workload")
	cmd.Flags().StringVarP(&OwnerReference, "owner-references", "", "",
		"The owner reference of all injectied resources")
	cmd.Flags().BoolVarP(&ApplyResources, "apply-resources", "", false,
		"Whether to apply the resources including the etcd cluster and the rpc service "+
			"if you enable this flag, the etcd cluster and the rpc service will be created during "+
			"the injection")
	cmd.Flags().StringVarP(&OutputFormat, "output", "o", "yaml",
		"The output format of the command, support yaml and json")
	cmd.Flags().StringVarP(&VineyardSecurityContext, "securityContext", "", "",
		"the json string of security context of vineyard sidecar container")
	cmd.Flags().StringVarP(&VineyardVolume, "volume", "", "",
		"the json string of vineyard sidecar container volume")
	cmd.Flags().StringVarP(&VineyardVolumeMount, "volumeMount", "", "",
		"the json string of vineyard sidecar container volume mount")
}
