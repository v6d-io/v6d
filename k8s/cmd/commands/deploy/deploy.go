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
)

// deployCmd deploys all vineyard components on kubernetes
var deployCmd = &cobra.Command{
	Use:   "deploy",
	Short: "Deploy the vineyard components on kubernetes",
	Long: `Deploy a vineyard component on kubernetes.
For example:

# deploy the default vineyard cluster on kubernetes
vineyarctl -k /home/gsbot/.kube/config deploy vineyard-cluster

# deploy the vineyard operator on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config deploy operator

# deploy the cert-manager on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config deploy cert-manager

# deploy the vineyardd on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config deploy vineyardd`,
	Run: func(cmd *cobra.Command, args []string) {
	},
}

func NewDeployCmd() *cobra.Command {
	return deployCmd
}

func init() {
	deployCmd.AddCommand(NewDeployOperatorCmd())
	deployCmd.AddCommand(NewDeployCertManagerCmd())
	deployCmd.AddCommand(NewDeployVineyarddCmd())
	deployCmd.AddCommand(NewDeployVineyardClusterCmd())
}
