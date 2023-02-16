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
package create

import (
	"fmt"

	"github.com/spf13/cobra"
)

// createCmd creates serveral vineyard jobs on kubernetes
var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Create the vineyard jobs on kubernetes",
	Long: `Create a specific vineyard job on kubernetes.
For example:

# create the backup job on kubernetes
vineyarctl -k /home/gsbot/.kube/config create backup --vineyardd-name vineyardd-sample --vineyardd-namespace vineyard-system`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("create called")
	},
}

func NewCreateCmd() *cobra.Command {
	return createCmd
}

func init() {
	createCmd.AddCommand(NewCreateBackupCmd())
	createCmd.AddCommand(NewCreateRecoverCmd())
}
