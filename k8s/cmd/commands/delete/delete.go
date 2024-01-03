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
package delete

import (
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var deleteExample = util.Examples(`
	# delete the default vineyard cluster on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete

	# delete the default vineyard operator on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete operator

	# delete the default vineyardd on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyardd`)

// deleteCmd represents the delete command
var deleteCmd = &cobra.Command{
	Use:     "delete",
	Short:   "Delete the vineyard components from kubernetes",
	Example: deleteExample,
}

func NewDeleteCmd() *cobra.Command {
	return deleteCmd
}

func init() {
	deleteCmd.AddCommand(NewDeleteOperatorCmd())
	deleteCmd.AddCommand(NewDeleteVineyarddCmd())
	deleteCmd.AddCommand(NewDeleteVineyardClusterCmd())

	deleteCmd.AddCommand(NewDeleteBackupCmd())
	deleteCmd.AddCommand(NewDeleteRecoverCmd())
	deleteCmd.AddCommand(NewDeleteVineyardDeploymentCmd())
	deleteCmd.AddCommand(NewDeleteOperationCmd())
	deleteCmd.AddCommand(NewDeleteCSIDriverCmd())
}
