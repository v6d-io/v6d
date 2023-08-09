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

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var deleteRecoverExample = util.Examples(`
	# delete the default recover job on kubernetes
	vineyardctl delete recover`)

// deleteRecoverCmd deletes the vineyard operator on kubernetes
var deleteRecoverCmd = &cobra.Command{
	Use:     "recover",
	Short:   "Delete the recover job from kubernetes",
	Example: deleteRecoverExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		log.Info("deleting Recover cr")
		recover := &v1alpha1.Recover{}
		if err := util.Delete(client, types.NamespacedName{
			Name:      flags.RecoverName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		}, recover); err != nil {
			log.Fatal(err, "failed to delete recover job")
		}
		log.Info("Recover Job is deleted.")
	},
}

func NewDeleteRecoverCmd() *cobra.Command {
	return deleteRecoverCmd
}

func init() {
	flags.ApplyRecoverNameOpts(deleteRecoverCmd)
}
