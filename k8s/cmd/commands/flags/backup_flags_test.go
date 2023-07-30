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

/*func TestApplyBackupNameOpts(t *testing.T) {
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
			ApplyBackupNameOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyBackupNameOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyBackupNameOpts(cmd)

	t.Run("DefaultBackupName", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认备份名称是否正确设置
		if BackupName != "vineyard-backup" {
			t.Errorf("Default backup name is incorrect, got: %s, want: vineyard-backup", BackupName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomBackupName", func(t *testing.T) {
		// 设置自定义备份名称参数
		cmd.SetArgs([]string{"--backup-name", "custom-backup"})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义备份名称是否正确解析和设置
		if BackupName != "custom-backup" {
			t.Errorf("Custom backup name is incorrect, got: %s, want: custom-backup", BackupName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

/*func TestApplyBackupCommonOpts(t *testing.T) {
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
			ApplyBackupCommonOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyBackupCommonOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyBackupCommonOpts(cmd)

	t.Run("DefaultValues", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查备份路径是否正确设置为默认值
		if BackupOpts.BackupPath != "" {
			t.Errorf("Default backup path is incorrect, got: %s, want: empty string", BackupOpts.BackupPath)
		}

		// 检查PV和PVC是否正确设置为默认值
		if BackupPVandPVC != "" {
			t.Errorf("Default PV and PVC value is incorrect, got: %s, want: empty string", BackupPVandPVC)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomValues", func(t *testing.T) {
		// 设置自定义参数值
		cmd.SetArgs([]string{"--path", "/backup", "--pv-pvc-spec", "pv-pvc-spec-value"})

		// 执行命令
		err := cmd.Execute()

		// 检查备份路径是否正确解析和设置
		if BackupOpts.BackupPath != "/backup" {
			t.Errorf("Custom backup path is incorrect, got: %s, want: /backup", BackupOpts.BackupPath)
		}

		// 检查PV和PVC是否正确解析和设置
		if BackupPVandPVC != "pv-pvc-spec-value" {
			t.Errorf("Custom PV and PVC value is incorrect, got: %s, want: pv-pvc-spec-value", BackupPVandPVC)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

/*func TestApplyCreateBackupOpts(t *testing.T) {
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
			ApplyCreateBackupOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyCreateBackupOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyCreateBackupOpts(cmd)

	t.Run("DefaultValues", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查备份名称相关选项的默认值
		if BackupName != "vineyard-backup" {
			t.Errorf("Default backup name is incorrect, got: %s, want: vineyard-backup", BackupName)
		}
		if BackupOpts.BackupPath != "" {
			t.Errorf("Default backup path is incorrect, got: %s, want: empty string", BackupOpts.BackupPath)
		}
		if BackupPVandPVC != "" {
			t.Errorf("Default PV and PVC value is incorrect, got: %s, want: empty string", BackupPVandPVC)
		}

		// 检查vineyardd选项的默认值
		if BackupOpts.VineyarddName != "" {
			t.Errorf("Default vineyardd name is incorrect, got: %s, want: empty string", BackupOpts.VineyarddName)
		}
		if BackupOpts.VineyarddNamespace != "" {
			t.Errorf("Default vineyardd namespace is incorrect, got: %s, want: empty string", BackupOpts.VineyarddNamespace)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomValues", func(t *testing.T) {
		// 设置自定义参数值
		cmd.SetArgs([]string{
			"--backup-name", "custom-backup",
			"--path", "/backup",
			"--pv-pvc-spec", "pv-pvc-spec-value",
			"--vineyardd-name", "vineyardd-1",
			"--vineyardd-namespace", "namespace-1",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查备份名称相关选项的自定义值
		if BackupName != "custom-backup" {
			t.Errorf("Custom backup name is incorrect, got: %s, want: custom-backup", BackupName)
		}
		if BackupOpts.BackupPath != "/backup" {
			t.Errorf("Custom backup path is incorrect, got: %s, want: /backup", BackupOpts.BackupPath)
		}
		if BackupPVandPVC != "pv-pvc-spec-value" {
			t.Errorf("Custom PV and PVC value is incorrect, got: %s, want: pv-pvc-spec-value", BackupPVandPVC)
		}

		// 检查vineyardd选项的自定义值
		if BackupOpts.VineyarddName != "vineyardd-1" {
			t.Errorf("Custom vineyardd name is incorrect, got: %s, want: vineyardd-1", BackupOpts.VineyarddName)
		}
		if BackupOpts.VineyarddNamespace != "namespace-1" {
			t.Errorf("Custom vineyardd namespace is incorrect, got: %s, want: namespace-1", BackupOpts.VineyarddNamespace)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

/*func TestApplyDeployBackupJobOpts(t *testing.T) {
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
			ApplyDeployBackupJobOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyDeployBackupJobOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyDeployBackupJobOpts(cmd)

	t.Run("DefaultValues", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查备份名称相关选项的默认值
		if BackupName != "vineyard-backup" {
			t.Errorf("Default backup name is incorrect, got: %s, want: vineyard-backup", BackupName)
		}
		if BackupOpts.BackupPath != "" {
			t.Errorf("Default backup path is incorrect, got: %s, want: empty string", BackupOpts.BackupPath)
		}
		if BackupPVandPVC != "" {
			t.Errorf("Default PV and PVC value is incorrect, got: %s, want: empty string", BackupPVandPVC)
		}

		// 检查其他选项的默认值
		if PVCName != "" {
			t.Errorf("Default PVC name is incorrect, got: %s, want: empty string", PVCName)
		}
		if VineyardDeploymentName != "" {
			t.Errorf("Default vineyard deployment name is incorrect, got: %s, want: empty string", VineyardDeploymentName)
		}
		if VineyardDeploymentNamespace != "" {
			t.Errorf("Default vineyard deployment namespace is incorrect, got: %s, want: empty string", VineyardDeploymentNamespace)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomValues", func(t *testing.T) {
		// 设置自定义参数值
		cmd.SetArgs([]string{
			"--backup-name", "custom-backup",
			"--path", "/backup",
			"--pv-pvc-spec", "pv-pvc-spec-value",
			"--pvc-name", "my-pvc",
			"--vineyard-deployment-name", "vineyard-1",
			"--vineyard-deployment-namespace", "namespace-1",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查备份名称相关选项的自定义值
		if BackupName != "custom-backup" {
			t.Errorf("Custom backup name is incorrect, got: %s, want: custom-backup", BackupName)
		}
		if BackupOpts.BackupPath != "/backup" {
			t.Errorf("Custom backup path is incorrect, got: %s, want: /backup", BackupOpts.BackupPath)
		}
		if BackupPVandPVC != "pv-pvc-spec-value" {
			t.Errorf("Custom PV and PVC value is incorrect, got: %s, want: pv-pvc-spec-value", BackupPVandPVC)
		}

		// 检查其他选项的自定义值
		if PVCName != "my-pvc" {
			t.Errorf("Custom PVC name is incorrect, got: %s, want: my-pvc", PVCName)
		}
		if VineyardDeploymentName != "vineyard-1" {
			t.Errorf("Custom vineyard deployment name is incorrect, got: %s, want: vineyard-1", VineyardDeploymentName)
		}
		if VineyardDeploymentNamespace != "namespace-1" {
			t.Errorf("Custom vineyard deployment namespace is incorrect, got: %s, want: namespace-1", VineyardDeploymentNamespace)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
