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

// Package start contains the start command of vineyard operator
package start

import (
	"github.com/spf13/cobra"
)

// startCmd starts the components of vineyard cluster such as controller, webhooks and scheduler
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the manager of vineyard operator",
	Long: `Start the manager of vineyard operator.
For example:

# start the manager with default configuration(Enable the controller, webhooks and scheduler)
vineyarctl start manager`,
}

func NewStartCmd() *cobra.Command {
	return startCmd
}

func init() {
	startCmd.AddCommand(NewStartManagerCmd())
}
