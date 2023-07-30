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
package util

import (
	"testing"

	"github.com/spf13/cobra"
)

/*func Test_AssertNoArgs(t *testing.T) {
	// 创建一个命令
	cmd := &cobra.Command{
		Use: "mycmd",
		Run: func(cmd *cobra.Command, args []string) {
			// 在运行函数中调用 AssertNoArgs 函数
			AssertNoArgs(cmd, args)
			log.Println("No positional arguments")
		},
	}

	// 执行命令并传递一些位置参数
	err := cmd.Execute()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}*/

func TestAssertNoArgs(t *testing.T) {
	cmd := &cobra.Command{
		Use: "mycmd",
	}
	type args struct {
		cmd  *cobra.Command
		args []string
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
		{
			name: "NoArgs",
			args: args{
				cmd:  cmd,
				args: []string{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			AssertNoArgs(tt.args.cmd, tt.args.args)
		})
	}
}

/*func Test_AssertNoArgsOrInput(t *testing.T) {
	// 创建一个命令
	cmd := &cobra.Command{
		Use: "mycmd",
		Run: func(cmd *cobra.Command, args []string) {
			// 在运行函数中调用 AssertNoArgsOrInput 函数
			AssertNoArgsOrInput(cmd, args)
			log.Println("No positional arguments or input")
		},
	}

	// 执行命令并传递一些位置参数
	err := cmd.Execute()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// 执行命令并传递输入为 "-" 的参数
	_, err = cmd.ExecuteC()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}*/

func TestAssertNoArgsOrInput(t *testing.T) {
	cmd := &cobra.Command{
		Use: "mycmd",
	}

	tests := []struct {
		name string
		args []string
	}{
		{
			name: "NoArgsOrInput",
			args: []string{},
		},
		{
			name: "WithInput",
			args: []string{"-"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			AssertNoArgsOrInput(cmd, tt.args)
		})
	}
}
