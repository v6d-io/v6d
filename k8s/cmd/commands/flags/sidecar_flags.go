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
	// WorkloadYaml is the yaml of workload
	WorkloadYaml string

	// SidecarOpts holds all configuration of sidecar Spec
	SidecarOpts v1alpha1.SidecarSpec
)

func ApplySidecarOpts(cmd *cobra.Command) {
	// setup the vineyard container configuration of vineyard sidecar
	ApplyVineyardContainerOpts(&SidecarOpts.VineyardConfig, "sidecar", cmd)
	// setup the metric container configuration of vineyard sidecar
	ApplyMetricContainerOpts(&SidecarOpts.MetricConfig, "sidecar", cmd)
	// setup the vineyard service configuration of vineyard sidecar
	ApplyServiceOpts(&SidecarOpts.Service, "sidecar", cmd)
	// setup the vineyard volumes if needed
	ApplyVolumeOpts(&SidecarOpts.Volume, "sidecar", cmd)
	cmd.Flags().IntVarP(&SidecarOpts.Replicas, "etcd-replicas", "", 1,
		"the number of etcd replicas")
	cmd.Flags().StringVarP(&WorkloadYaml, "file", "f", "", "The yaml of workload")
}
