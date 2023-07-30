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
package usage

import (
	"os"
	"testing"

	"github.com/spf13/cobra"
)

func TestFlattenCommands(t *testing.T) {
	// 创建一些命令和子命令
	rootCmd := &cobra.Command{Use: "root"}
	subCmd1 := &cobra.Command{Use: "sub1"}
	subCmd2 := &cobra.Command{Use: "sub2"}
	subSubCmd := &cobra.Command{Use: "subsub"}

	rootCmd.AddCommand(subCmd1, subCmd2)
	subCmd2.AddCommand(subSubCmd)

	// 调用 flattenCommands 函数
	var flattened []*cobra.Command
	flattenCommands(rootCmd, &flattened)

	// 验证是否正确地将所有命令展开到切片中
	expected := []*cobra.Command{rootCmd, subCmd1, subCmd2, subSubCmd}
	if len(flattened) != len(expected) {
		t.Errorf("Expected flattened commands length %d, but got %d", len(expected), len(flattened))
	}

	for i, cmd := range flattened {
		if cmd != expected[i] {
			t.Errorf("Expected command at index %d to be %v, but got %v", i, expected[i], cmd)
		}
	}
}

func TestGenerateReference(t *testing.T) {
	// 创建一些命令和子命令
	rootCmd := &cobra.Command{Use: "root"}
	subCmd1 := &cobra.Command{Use: "sub1"}
	subCmd2 := &cobra.Command{Use: "sub2"}
	subSubCmd := &cobra.Command{Use: "subsub"}

	rootCmd.AddCommand(subCmd1, subCmd2)
	subCmd2.AddCommand(subSubCmd)

	// 调用 GenerateReference 函数
	file := "reference.md"
	err := GenerateReference(rootCmd, file)
	if err != nil {
		t.Fatalf("Error generating reference: %v", err)
	}

	// 验证生成的参考文档文件是否存在
	if _, err := os.Stat(file); os.IsNotExist(err) {
		t.Errorf("Expected reference file %s to exist, but it doesn't", file)
	}

	// 清理测试生成的参考文档文件
	err = os.Remove(file)
	if err != nil {
		t.Errorf("Error cleaning up reference file: %v", err)
	}
}
