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
	"strings"

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
	"github.com/v6d-io/v6d/k8s/pkg/templates"
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

	vineyarddFileName []string = []string{
		"vineyardd/deployment.yaml",
		"vineyardd/service.yaml",
		"vineyardd/serviceaccount.yaml",
		"vineyardd/etcd-service.yaml",
		"vineyardd/spill-pv.yaml",
		"vineyardd/spill-pvc.yaml",
	}
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

var label string

func init() {
	flags.ApplyVineyarddOpts(deployVineyardDeploymentCmd)
	deployVineyardDeploymentCmd.Flags().
		StringVarP(&label, "label", "l", "", "label of the vineyardd")
}

func getStorage(q resource.Quantity) string {
	return q.String()
}

// VineyarddLabelSelector contains the label selector of vineyardd
var VineyarddLabelSelector []k8s.ServiceLabelSelector

func getVineyarddLabelSelector() []k8s.ServiceLabelSelector {
	return VineyarddLabelSelector
}

func parseLabel(l string) error {
	VineyarddLabelSelector = []k8s.ServiceLabelSelector{}
	str := []string{}
	if !strings.Contains(l, ",") {
		str = append(str, label)
	} else {
		str = append(str, strings.Split(l, ",")...)
	}

	for i := range str {
		kv := strings.Split(str[i], "=")
		if len(kv) != 2 {
			return errors.Errorf("invalid label: %s", str[i])
		}
		VineyarddLabelSelector = append(
			VineyarddLabelSelector,
			k8s.ServiceLabelSelector{Key: kv[0], Value: kv[1]},
		)
	}

	return nil
}

// EtcdConfig holds the configuration of etcd
var EtcdConfig k8s.EtcdConfig

func getEtcdConfig() k8s.EtcdConfig {
	return EtcdConfig
}

// GetObjectsFromTemplate gets kubernetes resources from template for vineyardd
func GetObjectsFromTemplate() ([]*unstructured.Unstructured, error) {
	objects := []*unstructured.Unstructured{}
	vineyardManifests, err := templates.GetFilesRecursive("vineyardd")
	if err != nil {
		return objects, errors.Wrap(err, "failed to get vineyardd manifests")
	}

	if label != "" {
		err = parseLabel(label)
		if err != nil {
			return objects, errors.Wrap(err, "failed to parse label")
		}
	}

	tmplFunc := map[string]interface{}{
		"getStorage":              getStorage,
		"getServiceLabelSelector": getVineyarddLabelSelector,
		"getEtcdConfig":           getEtcdConfig,
	}

	files := map[string]bool{}
	// add the vineyardd template files
	for _, f := range vineyarddFileName {
		files[f] = true
	}

	// build vineyardd
	vineyardd, err := BuildVineyardManifestFromInput()
	if err != nil {
		return objects, errors.Wrap(err, "failed to build vineyardd")
	}

	// process the vineyard socket
	v1alpha1.PreprocessVineyarddSocket(vineyardd)
	for _, f := range vineyardManifests {
		if _, ok := files[f]; !ok {
			continue
		}
		obj, err := util.RenderManifestAsObj(f, vineyardd, tmplFunc)
		if err != nil {
			return objects, err
		}
		if obj.GetName() != "" {
			objects = append(objects, obj)
		}
	}

	objs, err := util.BuildObjsFromEtcdManifests(&EtcdConfig, vineyardd.Namespace,
		vineyardd.Spec.Etcd.Replicas, vineyardd.Spec.VineyardConfig.Image, vineyardd,
		tmplFunc)
	if err != nil {
		return objects, errors.Wrap(err, "failed to build etcd objects")
	}
	for i := range objs {
		if objs[i].GetName() != "" {
			objects = append(objects, objs[i])
		}
	}
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
