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
package drydelete

import (
	"github.com/spf13/cobra"
)

// dryDeleteCmd delete the kubernetes resources without vineyard operator
var dryDeleteCmd = &cobra.Command{
	Use:   "drydelete",
	Short: "Drydelete  delete the vineyardd resources without vineyard operator",
	Long: `Drydelete delete the vineyardd resources without vineyard operator.
For example:

# delete the default vineyard resources in the vineyard-system namespace
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config drydelete vineyardd`,
	Run: func(cmd *cobra.Command, args []string) {
	},
}

func NewDryDeleteCmd() *cobra.Command {
	return dryDeleteCmd
}

func init() {
	dryDeleteCmd.AddCommand(NewDryDeleteVineyarddCmd())
}
