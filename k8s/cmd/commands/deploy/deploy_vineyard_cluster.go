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
package deploy

import (
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var deployVineyardClusterExample = util.Examples(`
	# deploy the default vineyard cluster on kubernetes
	vineyardctl deploy vineyard-cluster`)

// deployVineyardClusterCmd deploys the vineyard cluster on kubernetes
var deployVineyardClusterCmd = &cobra.Command{
	Use:     "vineyard-cluster",
	Short:   "Deploy the vineyard cluster from kubernetes",
	Example: deployVineyardClusterExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		// deploy vineyard operator
		NewDeployOperatorCmd().Run(cmd, args)

		// deploy vineyardd
		NewDeployVineyarddCmd().Run(cmd, args)

		log.Info("Vineyard Cluster is ready.")
	},
}

func NewDeployVineyardClusterCmd() *cobra.Command {
	return deployVineyardClusterCmd
}
