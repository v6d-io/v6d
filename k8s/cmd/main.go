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
package main

import (
	"flag"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/cobra/doc"
	"github.com/spf13/pflag"

	kubectlTemplate "k8s.io/kubectl/pkg/util/templates"
	// import as early as possible to introduce the "version" global flag
	_ "k8s.io/component-base/version/verflag"

	"github.com/v6d-io/v6d/k8s/cmd/commands/create"
	"github.com/v6d-io/v6d/k8s/cmd/commands/delete"
	"github.com/v6d-io/v6d/k8s/cmd/commands/deploy"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"

	"github.com/v6d-io/v6d/k8s/cmd/commands/manager"
	"github.com/v6d-io/v6d/k8s/cmd/commands/schedule"
	"github.com/v6d-io/v6d/k8s/cmd/commands/sidecar"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util/usage"
)

var (
	cmdLong = kubectlTemplate.LongDesc(`vineyardctl is the command-line 
	tool for working with the Vineyard Operator. It supports creating, 
	deleting and checking status of Vineyard Operator. It also supports 
	managing the vineyard relevant components such as vineyardd and pluggable
	drivers`)
)

var cmd = &cobra.Command{
	Use:     "vineyardctl [command]",
	Version: "v0.11.2",
	Short:   "vineyardctl is the command-line tool for working with the Vineyard Operator",
	Long:    cmdLong,
}

func init() {
	// rewrite the global "version" flag introduced in `verflag`
	flags.RemoveVersionFlag(pflag.CommandLine)

	cmd.InitDefaultHelpCmd()
	cmd.InitDefaultHelpFlag()
	cmd.InitDefaultVersionFlag()

	flags.ApplyGlobalFlags(cmd)

	// disable completion command
	cmd.CompletionOptions.DisableDefaultCmd = true

	cmd.AddCommand(create.NewCreateCmd())
	cmd.AddCommand(delete.NewDeleteCmd())
	cmd.AddCommand(deploy.NewDeployCmd())
	cmd.AddCommand(manager.NewManagerCmd())
	cmd.AddCommand(schedule.NewScheduleCmd())
	cmd.AddCommand(sidecar.NewInjectCmd())
}

func main() {
	setupGenDoc()
	tryDumpUsage()
	if err := cmd.Execute(); err != nil {
		util.ErrLogger.Fatalf("Failed to execute root command: %+v", err)
		os.Exit(-1)
	}
}

func tryDumpUsage() {
	cmd.FParseErrWhitelist.UnknownFlags = true
	if err := cmd.ParseFlags(os.Args); err != nil {
		_ = cmd.Usage()
		cmd.PrintErrf("\nError when parsing flags: %v\n", err)
		os.Exit(-1)
	}
	cmd.FParseErrWhitelist.UnknownFlags = false

	if flags.DumpUsage {
		cmd.SetUsageFunc(usage.UsageJson)
		if err := cmd.Usage(); err != nil {
			cmd.PrintErrf("\nError: %+v\n", err)
		}
		os.Exit(0)
	}
}
func setupGenDoc() {
	var gendoc = false
	flag.BoolVar(&gendoc, "gendoc", false,
		"Auto generate the documentation for the command line tool."+
			" The generated documentation will be written to the"+
			` "references.md" file under the current directory.`)
	flag.Parse()
	if gendoc {
		if err := genDoc(cmd, "references.md"); err != nil {
			util.ErrLogger.Fatalf("Failed to generate the documentation: %+v", err)
		}
	}

}

func genDoc(root *cobra.Command, file string) error {
	cmds := []*cobra.Command{}
	cList := []*cobra.Command{root}
	for len(cList) > 0 {
		c := cList[0]
		cList = cList[1:]
		cmds = append(cmds, c)
		if c.HasSubCommands() {
			cList = append(cList, c.Commands()...)
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

	f, err := os.OpenFile(file, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	w := io.Writer(f)

	for i := range cmds {
		cmds[i].DisableAutoGenTag = true
		if err := doc.GenMarkdownCustom(cmds[i], w, func(s string) string {
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

	defer f.Close()
	return nil
}
