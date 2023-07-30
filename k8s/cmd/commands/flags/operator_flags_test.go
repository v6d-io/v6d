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

/*func TestApplyOperatorOpts(t *testing.T) {
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
			ApplyOperatorOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyOperatorOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyOperatorOpts(cmd)

	t.Run("DefaultOperatorOpts", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认操作选项
		if OperatorVersion != "dev" {
			t.Errorf("Default operator version is incorrect, got: %s, want: 'dev'", OperatorVersion)
		}
		if KustomizeDir != "" {
			t.Errorf("Default kustomize dir is incorrect, got: %s, want: ''", KustomizeDir)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOperatorOpts", func(t *testing.T) {
		// 设置自定义操作选项
		cmd.SetArgs([]string{
			"--version", "1.0.0",
			"--local", "/path/to/kustomize/dir",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义操作选项
		if OperatorVersion != "1.0.0" {
			t.Errorf("Custom operator version is incorrect, got: %s, want: '1.0.0'", OperatorVersion)
		}
		if KustomizeDir != "/path/to/kustomize/dir" {
			t.Errorf("Custom kustomize dir is incorrect, got: %s, want: '/path/to/kustomize/dir'", KustomizeDir)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
