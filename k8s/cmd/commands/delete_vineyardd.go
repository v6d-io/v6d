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
package commands

import (
	"context"
	"log"

	"github.com/spf13/cobra"
	vineyardV1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
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
		if err := ValidateNoArgs("delete vineyardd", args); err != nil {
			log.Fatal("failed to validate delete vineyardd command args and flags: ", err)
		}

		kubeClient, err := getKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		vineyardd := &vineyardV1alpha1.Vineyardd{}
		if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: VineyarddName, Namespace: GetDefaultVineyardNamespace()},
			vineyardd); err != nil && !apierrors.IsNotFound(err) {
			log.Fatal("failed to get vineyardd: ", err)
		}

		if err := kubeClient.Delete(context.Background(), vineyardd); err != nil {
			log.Fatal("failed to delete vineyardd: ", err)
		}

		log.Println("Vineyardd is deleted.")
	},
}

func NewDeleteVineyarddCmd() *cobra.Command {
	return deleteVineyarddCmd
}

func init() {
	deleteVineyarddCmd.Flags().StringVarP(&VineyarddName, "name", "", "vineyardd-sample", "the name of vineyardd")
}
