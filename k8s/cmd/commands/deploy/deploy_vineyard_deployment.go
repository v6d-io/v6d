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
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

var vineyarddFileName []string = []string{
	"vineyardd/deployment.yaml",
	"vineyardd/service.yaml",
	"vineyardd/serviceaccount.yaml",
	"vineyardd/etcd-service.yaml",
	"vineyardd/spill-pv.yaml",
	"vineyardd/spill-pvc.yaml",
}

var etcdFileName []string = []string{
	"etcd/deployment.yaml",
	"etcd/service.yaml",
}

// deployVineyardDeploymentCmd build and deploy the yaml file of vineyardd from stdin or file
var deployVineyardDeploymentCmd = &cobra.Command{
	Use:   "vineyard-deployment",
	Short: "DeployVineyardDeployment builds and deploy the yaml file of vineyardd wihout vineyard operator",
	Long: `Builds and deploy the yaml file of vineyardd the vineyardd without vineyard operator. You could
deploy a customized vineyardd from stdin or file.

For example:

# deploy the default vineyard deployment on kubernetes
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyard-deployment

# deploy the vineyard deployment with customized image
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyard-deployment --image vineyardd:v0.12.2`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := cobra.NoArgs(cmd, args); err != nil {
			util.ErrLogger.Fatal(err)
		}
		client := util.KubernetesClient()

		if err := applyVineyarddFromTemplate(client); err != nil {
			util.ErrLogger.Fatal("failed to apply vineyardd resources from template: ", err)
		}

		util.InfoLogger.Println("vineyard cluster deployed successfully")
	},
}

func NewDeployVineyardDeploymentCmd() *cobra.Command {
	return deployVineyardDeploymentCmd
}

var label string

func init() {
	flags.ApplyVineyarddOpts(deployVineyardDeploymentCmd)
	deployVineyardDeploymentCmd.Flags().StringVarP(&label, "label", "l", "", "label of the vineyardd")
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
			return fmt.Errorf("invalid label: %s", str[i])
		}
		VineyarddLabelSelector = append(VineyarddLabelSelector, k8s.ServiceLabelSelector{Key: kv[0], Value: kv[1]})
	}

	return nil
}

// EtcdConfig holds the configuration of etcd
var EtcdConfig k8s.EtcdConfig

func getEtcdConfig() k8s.EtcdConfig {
	return EtcdConfig
}

// GetObjsFromTemplate gets kubernetes resources from template for vineyardd
func GetObjsFromTemplate() ([]*unstructured.Unstructured, error) {
	objs := []*unstructured.Unstructured{}
	t := templates.NewEmbedTemplate()
	vineyardManifests, err := t.GetFilesRecursive("vineyardd")
	if err != nil {
		return objs, fmt.Errorf("failed to get vineyardd manifests: %v", err)
	}

	etcdManifests, err := t.GetFilesRecursive("etcd")
	if err != nil {
		return objs, fmt.Errorf("failed to get etcd manifests: %v", err)
	}

	if label != "" {
		err = parseLabel(label)
		if err != nil {
			return objs, fmt.Errorf("failed to parse label: %v", err)
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
	// add the etcd template files
	for _, f := range etcdFileName {
		files[f] = true
	}

	// build vineyardd
	vineyardd, err := BuildVineyardManifest()
	if err != nil {
		return objs, fmt.Errorf("failed to build vineyardd: %v", err)
	}

	// process the vineyard socket
	v1alpha1.PreprocessVineyarddSocket(vineyardd)
	for _, f := range vineyardManifests {
		if _, ok := files[f]; !ok {
			continue
		}
		manifest, err := t.ReadFile(f)
		if err != nil {
			return objs, fmt.Errorf("failed to read manifest %s: %v", f, err)
		}

		obj := &unstructured.Unstructured{}
		_, _ = swckkube.LoadTemplate(string(manifest), vineyardd, tmplFunc, obj)
		if obj.GetKind() != "" {
			objs = append(objs, obj)
		}
	}

	// set up the etcd
	EtcdConfig.Namespace = vineyardd.Namespace
	etcdEndpoints := make([]string, 0, vineyardd.Spec.Etcd.Replicas)
	replicas := vineyardd.Spec.Etcd.Replicas
	for i := 0; i < replicas; i++ {
		etcdEndpoints = append(etcdEndpoints, fmt.Sprintf("etcd%v=http://etcd%v:2380", strconv.Itoa(i), strconv.Itoa(i)))
	}
	EtcdConfig.Endpoints = strings.Join(etcdEndpoints, ",")
	// the etcd is built in the vineyardd image
	EtcdConfig.Image = vineyardd.Spec.VineyardConfig.Image
	for i := 0; i < replicas; i++ {
		EtcdConfig.Rank = i
		for _, ef := range etcdManifests {
			manifest, err := t.ReadFile(ef)
			if err != nil {
				return objs, fmt.Errorf("failed to read manifest %s: %v", ef, err)
			}
			obj := &unstructured.Unstructured{}
			_, _ = swckkube.LoadTemplate(string(manifest), vineyardd, tmplFunc, obj)
			if obj.GetKind() != "" {
				objs = append(objs, obj)
			}
		}
	}
	return objs, nil
}

// applyVineyarddFromTemplate creates kubernetes resources from template fir
func applyVineyarddFromTemplate(c client.Client) error {
	objs, err := GetObjsFromTemplate()
	if err != nil {
		return fmt.Errorf("failed to get vineyardd resources from template: %v", err)
	}

	for _, o := range objs {
		if err := c.Get(context.Background(), types.NamespacedName{
			Name:      o.GetName(),
			Namespace: o.GetNamespace(),
		}, o); err != nil {
			if apierrors.IsNotFound(err) {
				if err := c.Create(context.TODO(), o); err != nil {
					return fmt.Errorf("failed to create object %s: %v", o.GetName(), err)
				}
			} else {
				return fmt.Errorf("failed to get object %s: %v", o.GetName(), err)
			}
		}
	}
	return nil
}
