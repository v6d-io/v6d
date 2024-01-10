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
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var deleteVineyardClusterExample = util.Examples(`
	# delete the default vineyard cluster on kubernetes
	vineyardctl delete vineyard-cluster`)

// deleteVineyardClusterCmd deletes the vineyard cluster on kubernetes
var deleteVineyardClusterCmd = &cobra.Command{
	Use:     "vineyard-cluster",
	Short:   "Delete the vineyard cluster from kubernetes",
	Example: deleteVineyardClusterExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		// delete vineyardd
		NewDeleteVineyarddCmd().Run(cmd, args)

		// delete vineyard operator
		NewDeleteOperatorCmd().Run(cmd, args)

		log.Info("Vineyard Cluster is deleted.")
	},
}

func NewDeleteVineyardClusterCmd() *cobra.Command {
	return deleteVineyardClusterCmd
}
