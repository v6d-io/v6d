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
	"time"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// deleteOperatorCmd deletes the vineyard operator on kubernetes
var deleteOperatorCmd = &cobra.Command{
	Use:   "operator",
	Short: "Delete the vineyard operator on kubernetes",
	Long: `Delete the vineyard operator on kubernetes. 
For example:

# delete the default vineyard operator in the vineyard-system namespace
vineyardctl delete operator

# delete the specific version of vineyard operator in the vineyard-system namespace
vineyardctl -n vineyard-system --kubeconfig /home/gsbot/.kube/config delete operator -v 0.12.2

# delete the vineyard operator from local kustomize dir in the vineyard-system namespace
vineyardctl -n vineyard-system --kubeconfig /home/gsbot/.kube/config delete operator --local ../config/default`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := cobra.NoArgs(cmd, args); err != nil {
			util.ErrLogger.Fatal(err)
		}

		scheme := runtime.NewScheme()
		if err := util.AddSchemes(scheme); err != nil {
			util.ErrLogger.Fatal("failed to add client scheme: ", err)
		}

		kubeClient, err := util.GetKubeClient(scheme)
		if err != nil {
			util.ErrLogger.Fatal("failed to get kubeclient: ", err)
		}

		operatorManifests, err := util.BuildKustomizeDir(util.GetKustomizeDir())
		if err != nil {
			util.ErrLogger.Fatal("failed to build kustomize dir", err)
		}

		if err := util.DeleteManifests(kubeClient, []byte(operatorManifests),
			flags.GetDefaultVineyardNamespace(), scheme); err != nil {
			util.ErrLogger.Fatal("failed to delete operator manifests: ", err)
		}

		waitOperatorDeleted(kubeClient)

		util.InfoLogger.Println("Vineyard Operator is deleted.")
	},
}

// wait for the vineyard operator to be deleted
func waitOperatorDeleted(c client.Client) {
	_ = wait.PollImmediate(1*time.Second, 300*time.Second, func() (bool, error) {
		deployment := &appsv1.Deployment{}
		err := c.Get(context.TODO(), types.NamespacedName{Name: "vineyard-controller-manager",
			Namespace: flags.GetDefaultVineyardNamespace()}, deployment)
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	})
}

func NewDeleteOperatorCmd() *cobra.Command {
	return deleteOperatorCmd
}

func init() {
	flags.NewOperatorOpts(deleteOperatorCmd)
}
