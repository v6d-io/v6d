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
)

// deleteCmd represents the delete command
var deleteCmd = &cobra.Command{
	Use:   "delete",
	Short: "Delete the vineyard components on kubernetes",
	Long: `Delete the vineyard components on kubernetes.  For example:

# delete the default vineyard cluster on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config delete

# delete the default vineyard operator on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config delete operator

# delete the default cert-manager on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config delete cert-manager

# delete the default vineyardd on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config delete vineyardd`,
	Run: func(cmd *cobra.Command, args []string) {
	},
}

func NewDeleteCmd() *cobra.Command {
	return deleteCmd
}

func init() {
	deleteCmd.AddCommand(NewDeleteCertManagerCmd())
	deleteCmd.AddCommand(NewDeleteOperatorCmd())
	deleteCmd.AddCommand(NewDeleteVineyarddCmd())
	deleteCmd.AddCommand(NewDeleteVineyardClusterCmd())

	deleteCmd.AddCommand(NewDeleteBackupCmd())
	deleteCmd.AddCommand(NewDeleteRecoverCmd())
}
