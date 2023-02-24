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
	"context"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/dryapply"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// dryDeleteVineyarddCmd delete the kubernetes resources without vineyard operator
var dryDeleteVineyarddCmd = &cobra.Command{
	Use:   "vineyardd",
	Short: "Drydelete vineyardd delete the vineyardd resources without vineyard operator",
	Long: `Drydelete vineyardd delete the vineyardd resources without vineyard operator.
For example:

# delete the default vineyard resources in the vineyard-system namespace
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config drydelete vineyardd

# delete the vineyard resources in the vineyard-system namespace with specific name
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config drydelete vineyardd --name vineyardd-0`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("drydelete vineyardd", args); err != nil {
			util.ErrLogger.Fatal("failed to validate drydelete vineyardd command args and flags: ", err,
				"the extra args are: ", args)
		}

		scheme, err := util.GetClientgoScheme()
		if err != nil {
			util.ErrLogger.Fatal("failed to get client-go scheme: ", err)
		}

		kubeclient, err := util.GetKubeClient(scheme)
		if err != nil {
			util.ErrLogger.Fatal("failed to get kube client: ", err)
		}

		if err := deleteVineyarddFromTemplate(kubeclient); err != nil {
			util.ErrLogger.Fatal("failed to delete vineyardd resources from template: ", err)
		}

		util.InfoLogger.Println("vineyard cluster deleted successfully")
	},
}

// deleteVineyarddFromTemplate creates kubernetes resources from template fir
func deleteVineyarddFromTemplate(c client.Client) error {
	objs, err := dryapply.GetObjsFromTemplate()
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

func NewDryDeleteVineyarddCmd() *cobra.Command {
	return dryDeleteVineyarddCmd
}

func init() {
	flags.NewVineyarddNameOpts(dryDeleteVineyarddCmd)
}
