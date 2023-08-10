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
package deploy

import (
	"context"

	"github.com/spf13/cobra"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployOperatorLong = util.LongDesc(`Deploy the vineyard operator on kubernetes.`)

	deployOperatorExample = util.Examples(`
	# deploy the vineyard operator on the 'vineyard-system' namespace
	vineyardctl deploy operator
	
	# deploy the vineyard operator on the existing namespace
	vineyardctl deploy operator -n my-custom-namespace
	
	# deploy the vineyard operator on the new namespace
	vineyardctl deploy operator -n a-new-namespace-name --create-namespace`)
)

// deployOperatorCmd deploys the vineyard operator on kubernetes
var deployOperatorCmd = &cobra.Command{
	Use:     "operator",
	Short:   "Deploy the vineyard operator on kubernetes",
	Long:    deployOperatorLong,
	Example: deployOperatorExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()
		util.CreateNamespaceIfNotExist(client)

		operatorManifests, err := util.BuildKustomizeInEmbedDir()
		if err != nil {
			log.Fatal(err, "failed to build kustomize dir")
		}

		log.Info("applying operator manifests")
		if err := util.ApplyManifests(client, operatorManifests,
			flags.GetDefaultVineyardNamespace()); err != nil {
			log.Fatal(err, "failed to apply operator manifests")
		}

		if flags.Wait {
			log.Info("waiting for the vineyard operator to be ready")
			if err := waitOperatorReady(client); err != nil {
				log.Fatal(err, "failed to wait operator ready")
			}
		}

		log.Info("Vineyard Operator is ready.")
	},
}

func NewDeployOperatorCmd() *cobra.Command {
	return deployOperatorCmd
}

// wait for the vineyard operator to be ready
func waitOperatorReady(c client.Client) error {
	return util.Wait(func() (bool, error) {
		deployment := &appsv1.Deployment{}
		if err := c.Get(context.TODO(), types.NamespacedName{
			Name:      "vineyard-controller-manager",
			Namespace: flags.GetDefaultVineyardNamespace(),
		}, deployment); err != nil {
			return false, err
		}

		if deployment.Status.ReadyReplicas == *deployment.Spec.Replicas {
			// get the vineyard operator pods
			podList := &corev1.PodList{}
			ready := true
			if err := c.List(context.TODO(), podList, client.InNamespace(flags.GetDefaultVineyardNamespace()),
				client.MatchingLabels{"control-plane": "controller-manager"}); err != nil {
				return false, err
			}
			for _, pod := range podList.Items {
				for _, condition := range pod.Status.ContainerStatuses {
					if !condition.Ready {
						ready = false
						break
					}
				}
			}
			return ready, nil
		}
		return false, nil
	})
}
