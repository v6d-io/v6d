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

	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/deploy"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

func TestDeleteVineyardDeploymentCmd(t *testing.T) {
	// deploy a vineyardd for later delete operation
	flags.KubeConfig = os.Getenv("KUBECONFIG")
	flags.VineyarddName = "test-vineyardd-deployment-name"
	flags.Namespace = "vineyard-system"
	flags.VineyarddOpts.Replicas = 3
	flags.VineyarddOpts.EtcdReplicas = 1
	flags.VineyarddOpts.Vineyard.Image = "vineyardcloudnative/vineyardd:latest"
	flags.VineyarddOpts.Vineyard.CPU = ""
	flags.VineyarddOpts.Vineyard.Memory = ""
	flags.VineyarddOpts.Service.Port = 9600
	flags.VineyarddOpts.Service.Type = "ClusterIP"
	flags.VineyarddOpts.Volume.PvcName = ""
	flags.VineyarddOpts.Vineyard.Size = "256Mi"
	c := util.KubernetesClient()
	deployVineyardDeploymentCmd := deploy.NewDeployVineyardDeploymentCmd()
	deployVineyardDeploymentCmd.Run(deployVineyardDeploymentCmd, []string{})

	// delete operation
	deleteVineyardDeploymentCmd.Run(deleteVineyardDeploymentCmd, []string{})

	objects, _ := deploy.GetVineyardDeploymentObjectsFromTemplate()

	// test if the vineyardd has been deleted
	for _, obj := range objects {
		err := c.Get(context.TODO(), client.ObjectKeyFromObject(obj), obj)
		if (err == nil) || !apierrors.IsNotFound(err) {
			log.Error(err, "failed to deleted object")
		}
	}
}
