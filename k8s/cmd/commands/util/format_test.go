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
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLongDesc(t *testing.T) {
	long := `
	This is a long description.
	It has multiple lines.
	`

	expected := "This is a long description.\nIt has multiple lines."

	result := LongDesc(long)
	assert.Equal(t, expected, result, "Long description should be trimmed and formatted correctly")
}

func TestExamples(t *testing.T) {
	// 创建一段需要格式化的示例文本
	examples := "  Example 1\n  Example 2\n"

	// 调用 Examples 函数
	formatted := Examples(examples)

	// 验证结果是否符合预期
	expected := "  Example 1\n	Example 2\n"

	// 使用正则表达式替换连续的空白字符为单个空格
	re := regexp.MustCompile(`\s+`)
	expected = re.ReplaceAllString(strings.TrimRight(expected, "\n "), " ") + "\n"
	formatted = re.ReplaceAllString(strings.TrimRight(formatted, "\n "), " ") + "\n"
	if formatted != expected {
		t.Errorf("Expected:\n%s\n\nGot:\n%s", expected, formatted)
	}
}

func TestFormatterTrim(t *testing.T) {
	// 测试数据和预期结果
	testCases := []struct {
		input    string
		expected string
	}{
		{"  Hello, World!  ", "Hello, World!"},
		{"\nHello, World!\n", "Hello, World!"},
		{"\tHello, World!\t", "Hello, World!"},
		{"Hello, World!", "Hello, World!"},
	}

	for _, testCase := range testCases {
		// 创建一个 formatter 对象
		f := formatter{testCase.input}

		// 调用 trim 方法
		f = f.trim()

		// 检查结果是否符合预期
		if f.string != testCase.expected {
			t.Errorf("Expected '%s', but got '%s'", testCase.expected, f.string)
		}
	}
}

func TestFormatterTab(t *testing.T) {
	// 创建一个 formatter，其字符串为 "example"
	f := formatter{"example"}

	// 调用 tab 方法
	f = f.tab()

	// 检查结果是否以制表符开始
	if !strings.HasPrefix(f.string, "\t") {
		t.Errorf("Expected string to start with tab, but got %s", f.string)
	}
}

func TestFormatter_Doc(t *testing.T) {
	f := formatter{"\tThis is a test.\n\tThis is another test."}
	expected := "This is a test.\nThis is another test."
	if out := f.doc().string; out != expected {
		t.Errorf("Expected %q, but got %q", expected, out)
	}
}

func TestFormatter_Indent(t *testing.T) {
	f := formatter{"This is a test.\n\tThis is another test."}
	expected := "This is a test.\n  This is another test."
	if out := f.indent().string; out != expected {
		t.Errorf("Expected %q, but got %q", expected, out)
	}
}
