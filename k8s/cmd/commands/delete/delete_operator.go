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

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var deleteOperatorExample = util.Examples(`
	# delete the default vineyard operator in the vineyard-system namespace
	vineyardctl delete operator
	
	# delete the vineyard operator in a specific namespace
	vineyardctl delete operator -n <namespace>`)

// deleteOperatorCmd deletes the vineyard operator on kubernetes
var deleteOperatorCmd = &cobra.Command{
	Use:     "operator",
	Short:   "Delete the vineyard operator from kubernetes",
	Example: deleteOperatorExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		log.Info("building operator manifests from kustomize dir")
		operatorManifests, err := util.BuildKustomizeInEmbedDir()
		if err != nil {
			log.Fatal(err, "failed to build kustomize dir")
		}

		log.Info("deleting operator manifests")
		if err := util.DeleteManifests(client, operatorManifests,
			flags.GetDefaultVineyardNamespace()); err != nil {
			log.Fatal(err, "failed to delete operator manifests")
		}
		log.Info("Vineyard Operator is deleted.")
	},
}

func NewDeleteOperatorCmd() *cobra.Command {
	return deleteOperatorCmd
}
