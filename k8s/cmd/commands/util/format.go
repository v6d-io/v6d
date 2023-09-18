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
	"strings"
)

var Indentation = "  "

func LongDesc(long string) string {
	return formatter{long}.trim().doc().trim().string
}

func Examples(examples string) string {
	return formatter{examples}.trim().tab().indent().string
}

type formatter struct {
	string
}

func (f formatter) trim() formatter {
	f.string = strings.TrimSpace(f.string)
	return f
}

func (f formatter) tab() formatter {
	f.string = "	" + f.string
	return f
}

func (f formatter) doc() formatter {
	docString := []string{}
	for _, line := range strings.Split(f.string, "\n") {
		line = strings.Trim(line, "\t")
		docString = append(docString, line)
	}
	f.string = strings.Join(docString, "\n")
	return f
}

func (f formatter) indent() formatter {
	indentedLines := []string{}
	for _, line := range strings.Split(f.string, "\n") {
		t := strings.Count(line, "\t")
		// Replace the tab with two spaces
		trimmed := strings.TrimSpace(line)
		indentation := ""
		for i := 0; i < t; i++ {
			indentation += Indentation
		}
		indented := indentation + trimmed
		indentedLines = append(indentedLines, indented)
	}
	f.string = strings.Join(indentedLines, "\n")
	return f
}
