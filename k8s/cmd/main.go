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
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

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

var cmd = &cobra.Command{
	Use:     "vineyardctl [command]",
	Version: "v0.11.2",
	Short:   "vineyardctl is the command-line tool for working with the Vineyard Operator",
	Long: `vineyardctl is the command-line tool for working with the Vineyard Operator.
It supports creating, deleting and checking status of Vineyard Operator. It also
supports managing the vineyard relevant components such as vineyardd and pluggable
drivers`,
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
	if err := cmd.ParseFlags(os.Args); err != nil {
		_ = cmd.Usage()
		cmd.PrintErrf("\nError xxx: %v\n", err)
		os.Exit(-1)
	}

	if flags.DumpUsage {
		cmd.SetUsageFunc(usage.UsageJson)
		if err := cmd.Usage(); err != nil {
			cmd.PrintErrf("\nError: %+v\n", err)
		}
		os.Exit(0)
	}
	if err := cmd.Execute(); err != nil {
		util.ErrLogger.Fatalf("failed to execute root command: %+v", err)
		os.Exit(-1)
	}
}
