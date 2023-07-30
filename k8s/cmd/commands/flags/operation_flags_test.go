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

/*func TestApplyOperationName(t *testing.T) {
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
			ApplyOperationName(tt.args.cmd)
		})
	}
}*/

func TestApplyOperationName(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyOperationName(cmd)

	t.Run("DefaultOperationName", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认操作名称
		if OperationName != "" {
			t.Errorf("Default operation name is incorrect, got: %s, want: ''", OperationName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOperationName", func(t *testing.T) {
		// 设置自定义操作名称
		cmd.SetArgs([]string{
			"--name", "custom-operation",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义操作名称
		if OperationName != "custom-operation" {
			t.Errorf("Custom operation name is incorrect, got: %s, want: 'custom-operation'", OperationName)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

/*func TestApplyOperationOpts(t *testing.T) {
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
			ApplyOperationOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyOperationOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyOperationOpts(cmd)

	t.Run("DefaultOperationOpts", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认操作选项
		if OperationOpts.Name != "" {
			t.Errorf("Default operation name is incorrect, got: %s, want: ''", OperationOpts.Name)
		}
		if OperationOpts.Type != "" {
			t.Errorf("Default operation type is incorrect, got: %s, want: ''", OperationOpts.Type)
		}
		if OperationOpts.Require != "" {
			t.Errorf("Default operation require is incorrect, got: %s, want: ''", OperationOpts.Require)
		}
		if OperationOpts.Target != "" {
			t.Errorf("Default operation target is incorrect, got: %s, want: ''", OperationOpts.Target)
		}
		if OperationOpts.TimeoutSeconds != 600 {
			t.Errorf("Default operation timeout seconds is incorrect, got: %d, want: 600", OperationOpts.TimeoutSeconds)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomOperationOpts", func(t *testing.T) {
		// 设置自定义操作选项
		cmd.SetArgs([]string{
			"--kind", "assembly",
			"--type", "local",
			"--require", "job1",
			"--target", "job2",
			"--timeoutSeconds", "900",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义操作选项
		if OperationOpts.Name != "assembly" {
			t.Errorf("Custom operation name is incorrect, got: %s, want: 'assembly'", OperationOpts.Name)
		}
		if OperationOpts.Type != "local" {
			t.Errorf("Custom operation type is incorrect, got: %s, want: 'local'", OperationOpts.Type)
		}
		if OperationOpts.Require != "job1" {
			t.Errorf("Custom operation require is incorrect, got: %s, want: 'job1'", OperationOpts.Require)
		}
		if OperationOpts.Target != "job2" {
			t.Errorf("Custom operation target is incorrect, got: %s, want: 'job2'", OperationOpts.Target)
		}
		if OperationOpts.TimeoutSeconds != 900 {
			t.Errorf("Custom operation timeout seconds is incorrect, got: %d, want: 900", OperationOpts.TimeoutSeconds)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
