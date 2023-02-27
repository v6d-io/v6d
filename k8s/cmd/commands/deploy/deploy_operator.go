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
	"time"

	"github.com/spf13/cobra"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

// deployOperatorCmd deploys the vineyard operator on kubernetes
var deployOperatorCmd = &cobra.Command{
	Use:   "operator",
	Short: "Deploy the vineyard operator on kubernetes",
	Long: `Deploy the vineyard operator on kubernetes. You could specify a stable or
development version of the operator. The default kustomize dir is development
version from github repo. Also, you can install the stable version from github.
repo or a local kustomize dir. Besides, you can also deploy the vineyard operator in
an existing namespace. For example:

# install the development version in the vineyard-system namespace
# the default kustomize dir is the development version from github repo
# (https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false)
# and the default namespace is vineyard-system
vineyardctl deploy operator

# install the stable version from github repo in the test namespace
# the kustomize dir is
# (https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false&ref=v0.12.2)
vineyardctl -n test --kubeconfig $HOME/.kube/config deploy operator -v 0.12.2

# install the local kustomize dir
vineyardctl --kubeconfig $HOME/.kube/config deploy operator --local ../config/default`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := cobra.NoArgs(cmd, args); err != nil {
			util.ErrLogger.Fatal(err)
		}
		client := util.KubernetesClient()

		operatorManifests, err := util.BuildKustomizeDir(util.GetKustomizeDir())
		if err != nil {
			util.ErrLogger.Fatal("failed to build kustomize dir: ", err)
		}

		if err := util.ApplyManifests(client, []byte(operatorManifests),
			flags.GetDefaultVineyardNamespace()); err != nil {
			util.ErrLogger.Fatal("failed to apply operator manifests: ", err)
		}

		if err := waitOperatorReady(client); err != nil {
			util.ErrLogger.Fatal("failed to wait operator ready: ", err)
		}

		util.InfoLogger.Println("Vineyard Operator is ready.")
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
	return wait.PollImmediate(1*time.Second, 300*time.Second, func() (bool, error) {
		deployment := &appsv1.Deployment{}
		if err := c.Get(context.TODO(), types.NamespacedName{
			Name:      "vineyard-controller-manager",
			Namespace: flags.GetDefaultVineyardNamespace(),
		}, deployment); err != nil {
			return false, err
		}
		for _, cond := range deployment.Status.Conditions {
			if cond.Type == appsv1.DeploymentAvailable {
				return true, nil
			}
		}
		return false, nil
	})
}
