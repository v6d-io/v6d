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
	deployOperatorLong = util.LongDesc(`
	Deploy the vineyard operator on kubernetes. You could specify a
	stable or development version of the operator. The default
	kustomize dir is development version from github repo. Also, you
	can install the stable version from github repo or a local
	kustomize dir. Besides, you can also  deploy the vineyard
	operator in an existing namespace.`)

	deployOperatorExample = util.Examples(`
	# install the development version in the vineyard-system namespace
	# the default kustomize dir is the development version from github repo
	# (https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false)
	# and the default namespace is vineyard-system
	# wait for the vineyard operator to be ready(default option)
	vineyardctl deploy operator

	# not to wait for the vineyard operator to be ready
	vineyardctl deploy operator --wait=false

	# install the stable version from github repo in the test namespace
	# the kustomize dir is
	# (https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false&ref=v0.12.2)
	vineyardctl -n test --kubeconfig $HOME/.kube/config deploy operator -v 0.12.2

	# install the local kustomize dir
	vineyardctl --kubeconfig $HOME/.kube/config deploy operator --local ../config/default`)
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

		operatorManifests, err := util.BuildKustomizeInDir(util.GetKustomizeDir())
		if err != nil {
			log.Fatal(err, "failed to build kustomize dir")
		}

		if err := util.ApplyManifests(client, operatorManifests,
			flags.GetDefaultVineyardNamespace()); err != nil {
			log.Fatal(err, "failed to apply operator manifests")
		}

		if flags.Wait {
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

func init() {
	flags.ApplyOperatorOpts(deployOperatorCmd)
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
