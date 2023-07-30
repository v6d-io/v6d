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

/*func TestApplySchedulerWorkflowOpts(t *testing.T) {
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
			ApplySchedulerWorkflowOpts(tt.args.cmd)
		})
	}
}*/

func TestApplySchedulerWorkflowOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplySchedulerWorkflowOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if WorkflowFile != "" {
			t.Errorf("Default workflow file path is incorrect, got: %s, want: ''", WorkflowFile)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOptions", func(t *testing.T) {
		// 设置自定义选项
		cmd.SetArgs([]string{
			"--file", "/path/to/workflow.yaml",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义选项
		if WorkflowFile != "/path/to/workflow.yaml" {
			t.Errorf("Custom workflow file path is incorrect, got: %s, want: '/path/to/workflow.yaml'", WorkflowFile)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
