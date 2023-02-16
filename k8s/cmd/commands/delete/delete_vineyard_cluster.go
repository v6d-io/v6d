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
	"log"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

// deleteVineyardClusterCmd deletes the vineyard cluster on kubernetes
var deleteVineyardClusterCmd = &cobra.Command{
	Use:   "vineyard-cluster",
	Short: "Delete the vineyard cluster on kubernetes",
	Long: `Delete the vineyardd on kubernetes. You could delete the vineyardd cluster
on kubernetes quickly. 

For example:

# delete the default vineyard cluster on kubernetes
vineyardctl delete vineyard-cluster`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("deploy vineyardd", args); err != nil {
			log.Fatal("failed to validate delete vineyard-cluster command args and flags: ", err)
		}

		// delete vineyardd
		NewDeleteVineyarddCmd().Run(cmd, args)

		// delete vineyard operator
		NewDeleteOperatorCmd().Run(cmd, args)

		// delete cert-manager
		NewDeleteCertManagerCmd().Run(cmd, args)

		log.Println("Vineyard Cluster is deleted.")
	},
}

func NewDeleteVineyardClusterCmd() *cobra.Command {
	return deleteVineyardClusterCmd
}
