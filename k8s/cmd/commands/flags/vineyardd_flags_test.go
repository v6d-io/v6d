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
	"testing"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
)

/*func TestApplyVineyardContainerOpts(t *testing.T) {
	type args struct {
		c      *v1alpha1.VineyardConfig
		prefix string
		cmd    *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyVineyardContainerOpts(tt.args.c, tt.args.prefix, tt.args.cmd)
		})
	}
}

func TestApplyServiceOpts(t *testing.T) {
	type args struct {
		s      *v1alpha1.ServiceConfig
		prefix string
		cmd    *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyServiceOpts(tt.args.s, tt.args.prefix, tt.args.cmd)
		})
	}
}

func TestApplyVolumeOpts(t *testing.T) {
	type args struct {
		v      *v1alpha1.VolumeConfig
		prefix string
		cmd    *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyVolumeOpts(tt.args.v, tt.args.prefix, tt.args.cmd)
		})
	}
}

func TestApplyMetricContainerOpts(t *testing.T) {
	type args struct {
		m      *v1alpha1.MetricConfig
		prefix string
		cmd    *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyMetricContainerOpts(tt.args.m, tt.args.prefix, tt.args.cmd)
		})
	}
}

func TestApplyPluginImageOpts(t *testing.T) {
	type args struct {
		cmd *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyPluginImageOpts(tt.args.cmd)
		})
	}
}

func TestApplyVineyarddNameOpts(t *testing.T) {
	type args struct {
		cmd *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyVineyarddNameOpts(tt.args.cmd)
		})
	}
}

func TestApplyVineyarddOpts(t *testing.T) {
	type args struct {
		cmd *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyVineyarddOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyVineyardContainerOpts(t *testing.T) {
	cmd := &cobra.Command{}
	vineyardConfig := &v1alpha1.VineyardConfig{}
	ApplyVineyardContainerOpts(vineyardConfig, "prefix", cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if vineyardConfig.Image != "vineyardcloudnative/vineyardd:latest" {
			t.Errorf("Default vineyardd image is incorrect, got: %s, want: 'vineyardcloudnative/vineyardd:latest'", vineyardConfig.Image)
		}
		if vineyardConfig.ImagePullPolicy != "IfNotPresent" {
			t.Errorf("Default vineyardd image pull policy is incorrect, got: %s, want: 'IfNotPresent'", vineyardConfig.ImagePullPolicy)
		}
		if vineyardConfig.SyncCRDs != true {
			t.Errorf("Default vineyardd sync CRDs is incorrect, got: %v, want: true", vineyardConfig.SyncCRDs)
		}
		if vineyardConfig.Socket != DefaultVineyardSocket {
			t.Errorf("Default vineyardd socket is incorrect, got: %s, want: '%s'", vineyardConfig.Socket, DefaultVineyardSocket)
		}
		if vineyardConfig.Size != "" {
			t.Errorf("Default vineyardd size is incorrect, got: %s, want: ''", vineyardConfig.Size)
		}
		if vineyardConfig.ReserveMemory != false {
			t.Errorf("Default vineyardd reserve memory is incorrect, got: %v, want: false", vineyardConfig.ReserveMemory)
		}
		if vineyardConfig.StreamThreshold != 80 {
			t.Errorf("Default vineyardd stream threshold is incorrect, got: %d, want: 80", vineyardConfig.StreamThreshold)
		}
		if len(VineyardContainerEnvs) != 0 {
			t.Errorf("Default vineyardd container envs is incorrect, got: %v, want: empty slice", VineyardContainerEnvs)
		}
		if vineyardConfig.Spill.Name != "" {
			t.Errorf("Default vineyardd spill name is incorrect, got: %s, want: ''", vineyardConfig.Spill.Name)
		}
		if vineyardConfig.Spill.Path != "" {
			t.Errorf("Default vineyardd spill path is incorrect, got: %s, want: ''", vineyardConfig.Spill.Path)
		}
		if vineyardConfig.Spill.SpillLowerRate != "0.3" {
			t.Errorf("Default vineyardd spill lower rate is incorrect, got: %s, want: '0.3'", vineyardConfig.Spill.SpillLowerRate)
		}
		if vineyardConfig.Spill.SpillUpperRate != "0.8" {
			t.Errorf("Default vineyardd spill upper rate is incorrect, got: %s, want: '0.8'", vineyardConfig.Spill.SpillUpperRate)
		}
		if VineyardSpillPVandPVC != "" {
			t.Errorf("Default vineyardd spill PV and PVC spec is incorrect, got: %s, want: ''", VineyardSpillPVandPVC)
		}
		if vineyardConfig.Memory != "" {
			t.Errorf("Default vineyardd memory is incorrect, got: %s, want: ''", vineyardConfig.Memory)
		}
		if vineyardConfig.CPU != "" {
			t.Errorf("Default vineyardd CPU is incorrect, got: %s, want: ''", vineyardConfig.CPU)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyServiceOpts(t *testing.T) {
	cmd := &cobra.Command{}
	serviceConfig := &v1alpha1.ServiceConfig{}
	ApplyServiceOpts(serviceConfig, "prefix", cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if serviceConfig.Type != "ClusterIP" {
			t.Errorf("Default service type is incorrect, got: %s, want: 'ClusterIP'", serviceConfig.Type)
		}
		if serviceConfig.Port != 9600 {
			t.Errorf("Default service port is incorrect, got: %d, want: 9600", serviceConfig.Port)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyVolumeOpts(t *testing.T) {
	cmd := &cobra.Command{}
	volumeConfig := &v1alpha1.VolumeConfig{}
	ApplyVolumeOpts(volumeConfig, "prefix", cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if volumeConfig.PvcName != "" {
			t.Errorf("Default PVC name is incorrect, got: %s, want: ''", volumeConfig.PvcName)
		}
		if volumeConfig.MountPath != "" {
			t.Errorf("Default mount path is incorrect, got: %s, want: ''", volumeConfig.MountPath)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyMetricContainerOpts(t *testing.T) {
	cmd := &cobra.Command{}
	metricConfig := &v1alpha1.MetricConfig{}
	ApplyMetricContainerOpts(metricConfig, "prefix", cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if metricConfig.Enable != false {
			t.Errorf("Default metric enable is incorrect, got: %v, want: false", metricConfig.Enable)
		}
		if metricConfig.Image != "vineyardcloudnative/vineyard-grok-exporter:latest" {
			t.Errorf("Default metric image is incorrect, got: %s, want: 'vineyardcloudnative/vineyard-grok-exporter:latest'", metricConfig.Image)
		}
		if metricConfig.ImagePullPolicy != "IfNotPresent" {
			t.Errorf("Default metric image pull policy is incorrect, got: %s, want: 'IfNotPresent'", metricConfig.ImagePullPolicy)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyPluginImageOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyPluginImageOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if VineyarddOpts.PluginImage.BackupImage != "ghcr.io/v6d-io/v6d/backup-job" {
			t.Errorf("Default backup image is incorrect, got: %s, want: 'ghcr.io/v6d-io/v6d/backup-job'", VineyarddOpts.PluginImage.BackupImage)
		}
		if VineyarddOpts.PluginImage.RecoverImage != "ghcr.io/v6d-io/v6d/recover-job" {
			t.Errorf("Default recover image is incorrect, got: %s, want: 'ghcr.io/v6d-io/v6d/recover-job'", VineyarddOpts.PluginImage.RecoverImage)
		}
		if VineyarddOpts.PluginImage.DaskRepartitionImage != "ghcr.io/v6d-io/v6d/dask-repartition" {
			t.Errorf("Default dask repartition image is incorrect, got: %s, want: 'ghcr.io/v6d-io/v6d/dask-repartition'", VineyarddOpts.PluginImage.DaskRepartitionImage)
		}
		if VineyarddOpts.PluginImage.LocalAssemblyImage != "ghcr.io/v6d-io/v6d/local-assembly" {
			t.Errorf("Default local assembly image is incorrect, got: %s, want: 'ghcr.io/v6d-io/v6d/local-assembly'", VineyarddOpts.PluginImage.LocalAssemblyImage)
		}
		if VineyarddOpts.PluginImage.DistributedAssemblyImage != "ghcr.io/v6d-io/v6d/distributed-assembly" {
			t.Errorf("Default distributed assembly image is incorrect, got: %s, want: 'ghcr.io/v6d-io/v6d/distributed-assembly'", VineyarddOpts.PluginImage.DistributedAssemblyImage)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyVineyarddNameOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyVineyarddNameOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if VineyarddName != "vineyardd-sample" {
			t.Errorf("Default Vineyardd name is incorrect, got: %s, want: 'vineyardd-sample'", VineyarddName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

}

func TestApplyVineyarddOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyVineyarddOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if VineyarddOpts.Replicas != 3 {
			t.Errorf("Default number of vineyardd replicas is incorrect, got: %d, want: 3", VineyarddOpts.Replicas)
		}
		if VineyarddOpts.EtcdReplicas != 1 {
			t.Errorf("Default number of etcd replicas is incorrect, got: %d, want: 1", VineyarddOpts.EtcdReplicas)
		}
		if VineyarddFile != "" {
			t.Errorf("Default vineyardd file path is incorrect, got: %s, want: ''", VineyarddFile)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
