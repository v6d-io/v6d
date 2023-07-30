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
)

/*func TestApplyRecoverNameOpts(t *testing.T) {
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
			ApplyRecoverNameOpts(tt.args.cmd)
		})
	}
}

func TestApplyCreateRecoverOpts(t *testing.T) {
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
			ApplyCreateRecoverOpts(tt.args.cmd)
		})
	}
}

func TestApplyDeployRecoverJobOpts(t *testing.T) {
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
			ApplyDeployRecoverJobOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyRecoverNameOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyRecoverNameOpts(cmd)

	t.Run("DefaultRecoverName", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认恢复作业名称
		if RecoverName != "vineyard-recover" {
			t.Errorf("Default recover job name is incorrect, got: %s, want: 'vineyard-recover'", RecoverName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomRecoverName", func(t *testing.T) {
		// 设置自定义恢复作业名称
		cmd.SetArgs([]string{
			"--recover-name", "custom-recover",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义恢复作业名称
		if RecoverName != "custom-recover" {
			t.Errorf("Custom recover job name is incorrect, got: %s, want: 'custom-recover'", RecoverName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyCreateRecoverOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyCreateRecoverOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if RecoverName != "vineyard-recover" {
			t.Errorf("Default recover job name is incorrect, got: %s, want: 'vineyard-recover'", RecoverName)
		}
		if BackupName != "vineyard-backup" {
			t.Errorf("Default backup job name is incorrect, got: %s, want: 'vineyard-backup'", BackupName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOptions", func(t *testing.T) {
		// 设置自定义选项
		cmd.SetArgs([]string{
			"--recover-name", "custom-recover",
			"--backup-name", "custom-backup",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义选项
		if RecoverName != "custom-recover" {
			t.Errorf("Custom recover job name is incorrect, got: %s, want: 'custom-recover'", RecoverName)
		}
		if BackupName != "custom-backup" {
			t.Errorf("Custom backup job name is incorrect, got: %s, want: 'custom-backup'", BackupName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

func TestApplyDeployRecoverJobOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyDeployRecoverJobOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if RecoverName != "vineyard-recover" {
			t.Errorf("Default recover job name is incorrect, got: %s, want: 'vineyard-recover'", RecoverName)
		}
		if RecoverPath != "" {
			t.Errorf("Default recover path is incorrect, got: %s, want: ''", RecoverPath)
		}
		if VineyardDeploymentName != "" {
			t.Errorf("Default vineyard deployment name is incorrect, got: %s, want: ''", VineyardDeploymentName)
		}
		if VineyardDeploymentNamespace != "" {
			t.Errorf("Default vineyard deployment namespace is incorrect, got: %s, want: ''", VineyardDeploymentNamespace)
		}
		if PVCName != "" {
			t.Errorf("Default PVC name is incorrect, got: %s, want: ''", PVCName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOptions", func(t *testing.T) {
		// 设置自定义选项
		cmd.SetArgs([]string{
			"--recover-name", "custom-recover",
			"--recover-path", "/path/to/recover",
			"--vineyard-deployment-name", "vineyard-deployment",
			"--vineyard-deployment-namespace", "vineyard-namespace",
			"--pvc-name", "pvc-name",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义选项
		if RecoverName != "custom-recover" {
			t.Errorf("Custom recover job name is incorrect, got: %s, want: 'custom-recover'", RecoverName)
		}
		if RecoverPath != "/path/to/recover" {
			t.Errorf("Custom recover path is incorrect, got: %s, want: '/path/to/recover'", RecoverPath)
		}
		if VineyardDeploymentName != "vineyard-deployment" {
			t.Errorf("Custom vineyard deployment name is incorrect, got: %s, want: 'vineyard-deployment'", VineyardDeploymentName)
		}
		if VineyardDeploymentNamespace != "vineyard-namespace" {
			t.Errorf("Custom vineyard deployment namespace is incorrect, got: %s, want: 'vineyard-namespace'", VineyardDeploymentNamespace)
		}
		if PVCName != "pvc-name" {
			t.Errorf("Custom PVC name is incorrect, got: %s, want: 'pvc-name'", PVCName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
