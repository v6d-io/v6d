/*
* This file is referred and modified from cobra project for flexibility,

	https://github.com/spf13/cobra/blob/main/doc/md_docs.go

 * which has the following license:

 //Copyright 2015 Red Hat Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

package usage

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"sort"
	"strings"
	"testing"

	"regexp"

	"github.com/spf13/cobra"
)

func TestHasSeeAlso(t *testing.T) {
	// 创建一个命令并设置它的父命令
	parentCmd := &cobra.Command{
		Use: "parent",
	}
	cmd := &cobra.Command{
		Use:   "test",
		Short: "Test command",
	}
	parentCmd.AddCommand(cmd)

	// 测试父命令的情况
	result := hasSeeAlso(cmd)
	if !result {
		t.Errorf("Expected hasSeeAlso(cmd) to be true, but got false")
	}

	// 测试没有父命令的情况
	result = hasSeeAlso(parentCmd)
	if result {
		t.Errorf("Expected hasSeeAlso(parentCmd) to be false, but got true")
	}

	// 测试有其他子命令的情况
	subCmd := &cobra.Command{
		Use:   "sub",
		Short: "Sub command",
	}
	cmd.AddCommand(subCmd)

	result = hasSeeAlso(cmd)
	if !result {
		t.Errorf("Expected hasSeeAlso(cmd) to be true, but got false")
	}

}

func TestByNameLen(t *testing.T) {
	// 创建一个包含命令的 byName 切片
	commands := byName{
		&cobra.Command{Use: "command1"},
		&cobra.Command{Use: "command2"},
		&cobra.Command{Use: "command3"},
	}

	// 调用 Len() 方法获取切片长度
	length := commands.Len()

	// 验证长度是否与预期相符
	expectedLength := 3
	if length != expectedLength {
		t.Errorf("Expected length: %d, got: %d", expectedLength, length)
	}
}

func TestByNameSwap(t *testing.T) {
	// 创建一个包含命令的 byName 切片
	commands := byName{
		&cobra.Command{Use: "command1"},
		&cobra.Command{Use: "command2"},
		&cobra.Command{Use: "command3"},
	}

	// 交换索引为 0 和 2 的元素
	commands.Swap(0, 2)

	// 验证交换后的切片顺序是否正确
	expectedCommands := byName{
		&cobra.Command{Use: "command3"},
		&cobra.Command{Use: "command2"},
		&cobra.Command{Use: "command1"},
	}

	for i := range commands {
		if commands[i].Use != expectedCommands[i].Use {
			t.Errorf("Expected command: %s, got: %s", expectedCommands[i].Use, commands[i].Use)
		}
	}
}

func TestByNameLess(t *testing.T) {
	// 创建一个包含命令的 byName 切片
	commands := byName{
		&cobra.Command{Use: "command3"},
		&cobra.Command{Use: "command1"},
		&cobra.Command{Use: "command2"},
	}

	// 对切片进行排序
	sort.Sort(commands)

	// 验证排序后的切片顺序是否正确
	expectedCommands := byName{
		&cobra.Command{Use: "command1"},
		&cobra.Command{Use: "command2"},
		&cobra.Command{Use: "command3"},
	}

	for i := range commands {
		if commands[i].Use != expectedCommands[i].Use {
			t.Errorf("Expected command: %s, got: %s", expectedCommands[i].Use, commands[i].Use)
		}
	}
}

func TestPrintOptions(t *testing.T) {
	// 创建一个缓冲区
	buf := new(bytes.Buffer)

	// 创建一个命令并添加一些选项
	cmd := &cobra.Command{
		Use:   "mycmd",
		Short: "My command",
	}
	cmd.Flags().String("option1", "", "Option 1 description")
	cmd.Flags().Bool("option2", false, "Option 2 description")
	//cmd.PersistentFlags().Int("option3", 0, "Option 3 description")

	// 调用 printOptions 函数
	err := printOptions(buf, "#", cmd, "mycmd")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// 验证输出是否符合预期
	expectedOutput := "## Options\n\n```\n  	  --option1 string   Option 1 description\n  	  --option2          Option 2 description\n```\n\n"
	actualOutput := buf.String()
	// 使用正则表达式替换连续的空白字符为单个空格
	re := regexp.MustCompile(`\s+`)
	expectedOutput = re.ReplaceAllString(expectedOutput, " ")
	actualOutput = re.ReplaceAllString(actualOutput, " ")

	if strings.Compare(expectedOutput, actualOutput) != 0 {
		t.Errorf("Expected output:\n%s\n\nActual output:\n%s", expectedOutput, actualOutput)
		fmt.Println(expectedOutput)
		fmt.Println(actualOutput)
	}
}

func TestGenMarkdown(t *testing.T) {
	// 创建一个命令
	cmd := &cobra.Command{
		Use:   "mycmd",
		Short: "My command",
	}

	// 创建一个缓冲区
	buf := new(bytes.Buffer)

	// 调用 GenMarkdown 函数
	err := GenMarkdown(cmd, buf)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// 验证输出是否非空
	if buf.Len() == 0 {
		t.Error("Expected non-empty output, got empty")
	}
}

func TestGenMarkdownCustom(t *testing.T) {
	// 创建一个命令
	cmd := &cobra.Command{
		Use:   "mycmd",
		Short: "My command",
		Long:  "This is a long description.",
	}
	cmd.SetUsageTemplate("Usage: mycmd [flags]")

	// 创建一个缓冲区
	buf := new(bytes.Buffer)

	// 定义 linkHandler 函数
	linkHandler := func(s string) string { return s }

	// 调用 GenMarkdownCustom 函数
	err := GenMarkdownCustom(cmd, buf, linkHandler)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// 从缓冲区读取输出内容
	output, err := ioutil.ReadAll(buf)
	if err != nil {
		t.Errorf("Failed to read output: %v", err)
	}

	// 验证输出是否包含预期内容
	expectedOutput := "# `mycmd`\n\n" +
		"My command\n\n" +
		"## Synopsis\n\n" +
		"This is a long description.\n\n" +
		"## Options\n\n" +
		"```\n" +
		"  -h, --help   help for mycmd\n" +
		"```\n\n" +
		"###### Auto generated by spf13/cobra on 11-Jul-2023"

	if !strings.Contains(string(output), expectedOutput) {
		t.Errorf("Expected output:\n%s\n\nActual output:\n%s", expectedOutput, output)
	}
}
