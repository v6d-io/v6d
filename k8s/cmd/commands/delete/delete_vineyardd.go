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
	"log"
	"time"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	vineyardV1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// deleteVineyarddCmd deletes the vineyardd cluster on kubernetes
var deleteVineyarddCmd = &cobra.Command{
	Use:   "vineyardd",
	Short: "Delete the vineyardd cluster on kubernetes",
	Long: `Delete the vineyardd cluster on kubernetes. 
For example:

# delete the default vineyardd cluster(vineyardd-sample) in the default namespace
vineyardctl delete vineyardd

# delete the specific vineyardd cluster in the vineyard-system namespace
vineyardctl -n vineyard-system delete vineyardd --name vineyardd-test`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("delete vineyardd", args); err != nil {
			log.Fatal("failed to validate delete vineyardd command args and flags: ", err)
		}

		kubeClient, err := util.GetKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		vineyardd := &vineyardV1alpha1.Vineyardd{}
		if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: flags.VineyarddName,
			Namespace: flags.GetDefaultVineyardNamespace()},
			vineyardd); err != nil && !apierrors.IsNotFound(err) {
			log.Fatal("failed to get vineyardd: ", err)
		}

		if err := kubeClient.Delete(context.Background(), vineyardd); err != nil {
			log.Fatal("failed to delete vineyardd: ", err)
		}

		waitVineyardDeleted(kubeClient, vineyardd)

		log.Println("Vineyardd is deleted.")
	},
}

// wait for the vineyardd to be deleted
func waitVineyardDeleted(c client.Client, vineyardd *v1alpha1.Vineyardd) {
	_ = wait.PollImmediate(1*time.Second, 300*time.Second, func() (bool, error) {
		err := c.Get(context.TODO(), types.NamespacedName{Name: vineyardd.Name,
			Namespace: vineyardd.Namespace}, vineyardd)
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	})
}

func NewDeleteVineyarddCmd() *cobra.Command {
	return deleteVineyarddCmd
}

func init() {
	flags.NewVineyarddNameOpts(deleteVineyarddCmd)
}
