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
package dryapply

import (
	"github.com/spf13/cobra"
)

// dryApplyCmd apply the kubernetes resources without vineyard operator
var dryApplyCmd = &cobra.Command{
	Use:   "dryapply",
	Short: "Dryapply the kubernetes resources without vineyard operator",
	Long: `Dryapply the kubernetes resources without vineyard operator.
For example:

# deploy the default vineyard on vineyard-system namespace
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config dryapply vineyardd

# deploy the vineyard on kubernetes with specific labels
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config dryapply vineyardd --labels app=vineyard,version=0.11.0`,
	Run: func(cmd *cobra.Command, args []string) {
	},
}

func NewDryApplyCmd() *cobra.Command {
	return dryApplyCmd
}

func init() {
	dryApplyCmd.AddCommand(NewDryApplyVineyarddCmd())
}
