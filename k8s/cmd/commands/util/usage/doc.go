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
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

func flattenCommands(cmd *cobra.Command, commands *[]*cobra.Command) {
	*commands = append(*commands, cmd)
	if cmd.HasSubCommands() {
		for _, c := range cmd.Commands() {
			flattenCommands(c, commands)
		}
	}
}

func GenerateReference(root *cobra.Command, file string) error {
	commands := []*cobra.Command{}
	flattenCommands(root, &commands)

	f, err := os.OpenFile(file, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	w := io.Writer(f)
	for _, cmd := range commands {
		cmd.DisableAutoGenTag = true
		if err := GenMarkdownCustom(cmd, w, func(s string) string {
			// the following code is to change the default
			// markdown file title to the link title
			s = strings.TrimSuffix(s, ".md")
			s = strings.ReplaceAll(s, "_", "-")
			s = "#" + s
			return s
		}); err != nil {
			return err
		}
	}
	return nil
}
