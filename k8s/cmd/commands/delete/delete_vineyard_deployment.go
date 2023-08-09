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
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/deploy"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var deleteVineyardDeploymentExample = util.Examples(`
	# delete the default vineyard deployment in the vineyard-system namespace
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyard-deployment

	# delete the vineyard deployment with specific name in the vineyard-system namespace
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyard-deployment \
		--name vineyardd-0`)

// deleteVineyardDeploymentCmd delete the vineyard deployment without vineyard operator
var deleteVineyardDeploymentCmd = &cobra.Command{
	Use:     "vineyard-deployment",
	Short:   "delete vineyard-deployment will delete the vineyard deployment without vineyard operator",
	Example: deleteVineyardDeploymentExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		log.Info("deleting vineyardd resources from template")
		if err := deleteVineyarddFromTemplate(client); err != nil {
			log.Fatal(err, "failed to delete vineyardd resources from template")
		}

		log.Info("vineyard cluster deleted successfully")
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
	objects, err := deploy.GetVineyardDeploymentObjectsFromTemplate()
	if err != nil {
		return errors.Wrap(err, "failed to get vineyardd resources from template")
	}

	for _, o := range objects {
		if err := util.Delete(c, client.ObjectKeyFromObject(o), o); err != nil {
			return errors.Wrapf(err, "failed to delete object: %s", o.GetName())
		}
	}
	return nil
}
