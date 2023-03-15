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
	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var createExample = util.Examples(`
	# create the backup job on kubernetes
	vineyardctl create backup --vineyardd-name vineyardd-sample --vineyardd-namespace vineyard-system

	# create the recover job on kubernetes
	vineyardctl create recover --backup-name vineyardd-sample -n vineyard-system

	# create the operation job on kubernetes
	vineyardctl create operation --name assembly \
		--type local \
		--require job1 \
		--target job2 \
		--timeoutSeconds 600`)

// createCmd creates several vineyard jobs on kubernetes
var createCmd = &cobra.Command{
	Use:     "create",
	Short:   "Create a vineyard jobs on kubernetes",
	Example: createExample,
}

func NewCreateCmd() *cobra.Command {
	return createCmd
}

func init() {
	createCmd.AddCommand(NewCreateBackupCmd())
	createCmd.AddCommand(NewCreateRecoverCmd())
	createCmd.AddCommand(NewCreateOperationCmd())
}
