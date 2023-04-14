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
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployVineyardDeploymentLong = util.LongDesc(`
	Builds and deploy the yaml file of vineyardd the vineyardd
	without vineyard operator. You could deploy a customized
	vineyardd from stdin or file.`)

	deployVineyardDeploymentExample = util.Examples(`
	# deploy the default vineyard deployment on kubernetes
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config \
	deploy vineyard-deployment

	# deploy the vineyard deployment with customized image
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config \
	deploy vineyard-deployment --image vineyardd:v0.12.2`)
)

// deployVineyardDeploymentCmd build and deploy the yaml file of vineyardd from stdin or file
var deployVineyardDeploymentCmd = &cobra.Command{
	Use: "vineyard-deployment",
	Short: "DeployVineyardDeployment builds and deploy the yaml file of " +
		"vineyardd without vineyard operator",
	Long:    deployVineyardDeploymentLong,
	Example: deployVineyardDeploymentExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()
		util.CreateNamespaceIfNotExist(client)

		if err := applyVineyarddFromTemplate(client); err != nil {
			log.Fatal(err, "failed to apply vineyardd resources from template")
		}

		log.Info("vineyard cluster deployed successfully")
	},
}

func NewDeployVineyardDeploymentCmd() *cobra.Command {
	return deployVineyardDeploymentCmd
}

func init() {
	flags.ApplyVineyarddOpts(deployVineyardDeploymentCmd)
}

func getStorage(q resource.Quantity) string {
	return q.String()
}

// EtcdConfig holds the configuration of etcd
var EtcdConfig k8s.EtcdConfig

func getEtcdConfig() k8s.EtcdConfig {
	return EtcdConfig
}

// GetObjectsFromTemplate gets kubernetes resources from template for vineyardd
func GetObjectsFromTemplate() ([]*unstructured.Unstructured, error) {
	objects := []*unstructured.Unstructured{}
	var err error

	tmplFunc := map[string]interface{}{
		"getStorage":    getStorage,
		"getEtcdConfig": getEtcdConfig,
	}

	// build vineyardd
	vineyardd, err := BuildVineyardManifestFromInput()
	if err != nil {
		return objects, errors.Wrap(err, "failed to build vineyardd")
	}

	// process the vineyard socket
	v1alpha1.PreprocessVineyarddSocket(vineyardd)

	objs, err := util.BuildObjsFromVineyarddManifests([]string{}, vineyardd, tmplFunc)
	if err != nil {
		return objects, errors.Wrap(err, "failed to build vineyardd objects")
	}
	objects = append(objects, objs...)

	podObjs, svcObjs, err := util.BuildObjsFromEtcdManifests(&EtcdConfig, vineyardd.Namespace,
		vineyardd.Spec.EtcdReplicas, vineyardd.Spec.Vineyard.Image, vineyardd,
		tmplFunc)
	if err != nil {
		return objects, errors.Wrap(err, "failed to build etcd objects")
	}
	objects = append(objects, append(podObjs, svcObjs...)...)
	return objects, nil
}

// applyVineyarddFromTemplate creates kubernetes resources from template fir
func applyVineyarddFromTemplate(c client.Client) error {
	objects, err := GetObjectsFromTemplate()
	if err != nil {
		return errors.Wrap(err, "failed to get vineyardd resources from template")
	}

	for _, o := range objects {
		if err := util.CreateIfNotExists(c, o); err != nil {
			return errors.Wrapf(err, "failed to create object %s", o.GetName())
		}
	}
	return nil
}
