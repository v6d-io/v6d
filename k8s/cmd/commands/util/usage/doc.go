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
	"github.com/spf13/cobra/doc"
)

func GenerateReference(root *cobra.Command, file string) error {
	commands := []*cobra.Command{}
	children := []*cobra.Command{root}
	for len(children) > 0 {
		c := children[0]
		children = children[1:]
		commands = append(commands, c)
		if c.HasSubCommands() {
			children = append(children, c.Commands()...)
		}
	}

	// Check if the file exists
	if _, err := os.Stat(file); os.IsNotExist(err) {
		// Create the file
		file, err := os.Create(file)
		if err != nil {
			return err
		}
		defer file.Close()
	}

	f, err := os.OpenFile(file, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	w := io.Writer(f)
	for i := range commands {
		commands[i].DisableAutoGenTag = true
		if err := doc.GenMarkdownCustom(commands[i], w, func(s string) string {
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
