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
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

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
	deploy vineyard-deployment --image vineyardcloudnative/vineyardd:v0.12.2`)

	// OwnerReference is the owner reference of all vineyard deployment resources
	OwnerReference string
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

		log.Info("waiting for vineyard deployment ready")
		if err := waitVineyardDeploymentReady(client); err != nil {
			log.Fatal(err, "failed to wait vineyard deployment for ready")
		}

		log.Info("vineyard cluster deployed successfully")
	},
}

func NewDeployVineyardDeploymentCmd() *cobra.Command {
	return deployVineyardDeploymentCmd
}

func init() {
	flags.ApplyVineyarddOpts(deployVineyardDeploymentCmd)
	deployVineyardDeploymentCmd.Flags().StringVarP(&OwnerReference, "owner-references",
		"", "", "The owner reference of all vineyard deployment resources")
}

// GetVineyardDeploymentObjectsFromTemplate gets kubernetes resources from template for vineyard-deployment
func GetVineyardDeploymentObjectsFromTemplate() ([]*unstructured.Unstructured, error) {
	objects := []*unstructured.Unstructured{}
	var etcdConfig k8s.EtcdConfig
	var err error

	tmplFunc := map[string]interface{}{
		"getStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
		"toYaml": func(v interface{}) string {
			bs, err := yaml.Marshal(v)
			if err != nil {
				log.Error(err, "failed to marshal object %v to yaml", v)
				return ""
			}
			return string(bs)
		},
		"indent": func(spaces int, s string) string {
			prefix := strings.Repeat(" ", spaces)
			return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
		},
	}

	// build vineyardd
	vineyardd, err := BuildVineyardManifestFromInput()
	if err != nil {
		return objects, errors.Wrap(err, "failed to build vineyardd")
	}

	podObjs, svcObjs, err := util.BuildObjsFromEtcdManifests(&etcdConfig, vineyardd.Name,
		vineyardd.Namespace, vineyardd.Spec.EtcdReplicas, vineyardd.Spec.Vineyard.Image, vineyardd,
		tmplFunc)
	if err != nil {
		return objects, errors.Wrap(err, "failed to build etcd objects")
	}
	objects = append(objects, append(podObjs, svcObjs...)...)

	// process the vineyard socket
	v1alpha1.PreprocessVineyarddSocket(vineyardd)

	objs, err := util.BuildObjsFromVineyarddManifests([]string{}, vineyardd, tmplFunc)
	if err != nil {
		return objects, errors.Wrap(err, "failed to build vineyardd objects")
	}
	objects = append(objects, objs...)

	return objects, nil
}

// applyVineyarddFromTemplate creates kubernetes resources from template fir
func applyVineyarddFromTemplate(c client.Client) error {
	objects, err := GetVineyardDeploymentObjectsFromTemplate()
	if err != nil {
		return errors.Wrap(err, "failed to get vineyardd resources from template")
	}
	deployment := &unstructured.Unstructured{}
	for _, o := range objects {
		if OwnerReference != "" {
			OwnerRefs, err := util.ParseOwnerRef(OwnerReference)
			if err != nil {
				return errors.Wrapf(err, "failed to parse owner reference %s", OwnerReference)
			}
			o.SetOwnerReferences(OwnerRefs)
		}
		if o.GetKind() == "Deployment" {
			deployment = o
			continue
		}
		waitETCDPodFunc := func(o *unstructured.Unstructured) bool {
			if o.GetKind() == "Pod" {
				pod := corev1.Pod{}
				if err := c.Get(context.TODO(), client.ObjectKey{
					Name:      o.GetName(),
					Namespace: o.GetNamespace(),
				}, &pod); err != nil {
					return false
				}
				if pod.Status.Phase == corev1.PodRunning {
					return true
				}
				return false
			}
			return true
		}
		if err := util.CreateIfNotExists(c, o, waitETCDPodFunc); err != nil {
			return errors.Wrapf(err, "failed to create object %s", o.GetName())
		}
	}
	// to reduce the time of waiting for the etcd cluster service ready in the vineyardd
	// wait the etcd cluster for ready here and create the vineyard deployment at last
	if err := util.CreateIfNotExists(c, deployment); err != nil {
		return errors.Wrapf(err, "failed to create object %s", deployment.GetName())
	}

	return nil
}

func waitVineyardDeploymentReady(c client.Client) error {
	return util.Wait(func() (bool, error) {
		name := client.ObjectKey{Name: flags.VineyarddName, Namespace: flags.Namespace}
		deployment := appsv1.Deployment{}
		if err := c.Get(context.TODO(), name, &deployment); err != nil {
			return false, errors.Wrap(err, "failed to get the vineyard-deployment")
		}
		if deployment.Status.ReadyReplicas == *deployment.Spec.Replicas {
			return true, nil
		}
		return false, nil
	})
}
