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
	"context"
	"fmt"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/deploy"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

// deleteVineyardDeploymentCmd delete the vineyard deployment without vineyard operator
var deleteVineyardDeploymentCmd = &cobra.Command{
	Use:   "vineyard-deployment",
	Short: "delete vineyard-deployment will delete the vineyard deployment without vineyard operator",
	Long: `delete vineyard-deployment will delete the vineyard deployment without vineyard operator.
For example:

# delete the default vineyard deployment in the vineyard-system namespace
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyard-deployment

# delete the vineyard deployment with specific name in the vineyard-system namespace
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyard-deployment --name vineyardd-0`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := cobra.NoArgs(cmd, args); err != nil {
			util.ErrLogger.Fatal(err)
		}
		client := util.KubernetesClient()

		if err := deleteVineyarddFromTemplate(client); err != nil {
			util.ErrLogger.Fatal("failed to delete vineyardd resources from template: ", err)
		}

		util.InfoLogger.Println("vineyard cluster deleted successfully")
	},
}

func NewDeleteVineyardDeploymentCmd() *cobra.Command {
	return deleteVineyardDeploymentCmd
}

func init() {
	flags.ApplyVineyarddNameOpts(deleteVineyardDeploymentCmd)
}

// deleteVineyarddFromTemplate creates kubernetes resources from template fir
func deleteVineyarddFromTemplate(c client.Client) error {
	objs, err := deploy.GetObjsFromTemplate()
	if err != nil {
		return fmt.Errorf("failed to get vineyardd resources from template: %v", err)
	}

	for _, o := range objs {
		if err := c.Get(context.Background(), client.ObjectKeyFromObject(o), o); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			} else {
				return fmt.Errorf("failed to get object %s: %v", o.GetName(), err)
			}
		}

		if err := c.Delete(context.Background(), o); err != nil {
			return fmt.Errorf("failed to delete object %s: %v", o.GetName(), err)
		}
	}
	return nil
}
