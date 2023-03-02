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
)

var (
	deployLong = util.LongDesc(`Deploy a vineyard component on kubernetes.`)

	deployExample = util.Examples(`
	# deploy the default vineyard cluster on kubernetes
	vineyardctl --kubeconfig $HOME/.kube/config deploy vineyard-cluster

	# deploy the vineyard operator on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy operator

	# deploy the cert-manager on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy cert-manager

	# deploy the vineyardd on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd`)
)

// deployCmd deploys all vineyard components on kubernetes
var deployCmd = &cobra.Command{
	Use:     "deploy",
	Short:   "Deploy the vineyard components on kubernetes",
	Long:    deployLong,
	Example: deployExample,
}

func NewDeployCmd() *cobra.Command {
	return deployCmd
}

func init() {
	deployCmd.AddCommand(newDeployOperatorCmd())
	deployCmd.AddCommand(newDeployCertManagerCmd())
	deployCmd.AddCommand(newDeployVineyarddCmd())
	deployCmd.AddCommand(newDeployVineyardClusterCmd())
	deployCmd.AddCommand(newDeployVineyardDeploymentCmd())
}