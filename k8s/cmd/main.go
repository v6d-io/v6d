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
	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/create"
	"github.com/v6d-io/v6d/k8s/cmd/commands/delete"
	"github.com/v6d-io/v6d/k8s/cmd/commands/deploy"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/manager"
	"github.com/v6d-io/v6d/k8s/cmd/commands/schedule"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var cmd = &cobra.Command{
	Use:   "vineyardctl [command]",
	Short: "vineyardctl is the command-line tool for working with the Vineyard Operator",
	Long: `vineyardctl is the command-line tool for working with the Vineyard Operator.
It supports creating, deleting and checking status of Vineyard Operator. It also
supports managing the vineyard relevant components such as vineyardd and pluggable
drivers`,
}

func init() {
	flags.ApplyGlobalFlags(cmd)

	// disable completion command
	cmd.CompletionOptions.DisableDefaultCmd = true

	cmd.AddCommand(deploy.NewDeployCmd())
	cmd.AddCommand(create.NewCreateCmd())
	cmd.AddCommand(delete.NewDeleteCmd())
	cmd.AddCommand(manager.NewManagerCmd())
	cmd.AddCommand(schedule.NewScheduleCmd())
}

func main() {
	if err := cmd.Execute(); err != nil {
		util.ErrLogger.Fatalf("failed to execute root command: %+v", err)
	}
}
