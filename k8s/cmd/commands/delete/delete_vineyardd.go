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
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/types"

	vineyardv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var deleteVineyarddExample = util.Examples(`
	# delete the default vineyardd cluster(vineyardd-sample) in the default namespace
	vineyardctl delete vineyardd

	# delete the specific vineyardd cluster in the vineyard-system namespace
	vineyardctl -n vineyard-system delete vineyardd --name vineyardd-test`)

// deleteVineyarddCmd deletes the vineyardd cluster on kubernetes
var deleteVineyarddCmd = &cobra.Command{
	Use:     "vineyardd",
	Short:   "Delete the vineyardd cluster from kubernetes",
	Example: deleteVineyarddExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		log.Info("deleting Vineyardd cluster")
		vineyardd := &vineyardv1alpha1.Vineyardd{}
		if err := util.Delete(client, types.NamespacedName{
			Name:      flags.VineyarddName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		}, vineyardd); err != nil {
			log.Fatal(err, "failed to delete vineyardd")
		}
		log.Info("Vineyardd is deleted.")
	},
}

func NewDeleteVineyarddCmd() *cobra.Command {
	return deleteVineyarddCmd
}

func init() {
	flags.ApplyVineyarddNameOpts(deleteVineyarddCmd)
}
