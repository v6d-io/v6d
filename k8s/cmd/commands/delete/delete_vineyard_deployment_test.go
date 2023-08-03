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
	"os"
	"testing"

	"github.com/v6d-io/v6d/k8s/cmd/commands/deploy"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestDeleteVineyardDeploymentCmd(t *testing.T) {
	flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
	flags.Namespace = "vineyard-system"
	c := util.KubernetesClient()

	deleteVineyardDeploymentCmd.Run(deleteVineyardDeploymentCmd, []string{})

	objects, _ := deploy.GetVineyardDeploymentObjectsFromTemplate()

	// test if the vineyardd has been deleted sucessfully
	for _, obj := range objects {
		if err := c.Get(context.TODO(), client.ObjectKeyFromObject(obj), obj); err == nil {
			log.Error(err, "failed to deleted object")
		}
	}
}
