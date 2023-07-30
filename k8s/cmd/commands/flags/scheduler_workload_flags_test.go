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

/*func TestApplySchedulerWorkloadOpts(t *testing.T) {
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
			ApplySchedulerWorkloadOpts(tt.args.cmd)
		})
	}
}*/

func TestApplySchedulerWorkloadOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplySchedulerWorkloadOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if Resource != "" {
			t.Errorf("Default resource JSON string is incorrect, got: %s, want: ''", Resource)
		}
		if VineyarddName != "vineyardd-sample" {
			t.Errorf("Default vineyardd name is incorrect, got: %s, want: 'vineyardd-sample'", VineyarddName)
		}
		if VineyarddNamespace != "vineyard-system" {
			t.Errorf("Default vineyardd namespace is incorrect, got: %s, want: 'vineyard-system'", VineyarddNamespace)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOptions", func(t *testing.T) {
		// 设置自定义选项
		cmd.SetArgs([]string{
			"--resource", `{"kind": "Deployment", "metadata": {"name": "my-deployment"}}`,
			"--vineyardd-name", "custom-vineyardd",
			"--vineyardd-namespace", "custom-namespace",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义选项
		if Resource != `{"kind": "Deployment", "metadata": {"name": "my-deployment"}}` {
			t.Errorf("Custom resource JSON string is incorrect, got: %s, want: '{\"kind\": \"Deployment\", \"metadata\": {\"name\": \"my-deployment\"}}'", Resource)
		}
		if VineyarddName != "custom-vineyardd" {
			t.Errorf("Custom vineyardd name is incorrect, got: %s, want: 'custom-vineyardd'", VineyarddName)
		}
		if VineyarddNamespace != "custom-namespace" {
			t.Errorf("Custom vineyardd namespace is incorrect, got: %s, want: 'custom-namespace'", VineyarddNamespace)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
