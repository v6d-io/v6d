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
	"time"

	"github.com/avast/retry-go"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployVineyarddLong = util.LongDesc(`
	Deploy the vineyardd on kubernetes. You could deploy a
	customized vineyardd from stdin or file.`)

	deployVineyarddExample = util.Examples(`
	# deploy the default vineyard on kubernetes
	# wait for the vineyardd to be ready(default option)
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd

	# not to wait for the vineyardd to be ready
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
		--wait=false

	# deploy the vineyardd from a yaml file
	vineyardctl --kubeconfig $HOME/.kube/config deploy vineyardd --file vineyardd.yaml

	# deploy the vineyardd with customized image
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
		--image vineyardd:v0.12.2

	# deploy the vineyardd with spill mechanism on persistent storage from json string
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
		--vineyardd.spill.config spill-path \
		--vineyardd.spill.path /var/vineyard/spill \
		--vineyardd.spill.pv-pvc-spec '{
			"pv-spec": {
				"capacity": {
					"storage": "1Gi"
				},
				"accessModes": [
					"ReadWriteOnce"
				],
				"storageClassName": "manual",
				"hostPath": {
					"path": "/var/vineyard/spill"
				}
			},
			"pvc-spec": {
				"storageClassName": "manual",
				"accessModes": [
					"ReadWriteOnce"
				],
				"resources": {
					"requests": {
					"storage": "512Mi"
					}
				}
			}
		}'

	# deploy the vineyardd with spill mechanism on persistent storage from yaml string
	vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
		--vineyardd.spill.config spill-path \
		--vineyardd.spill.path /var/vineyard/spill \
		--vineyardd.spill.pv-pvc-spec \
		'
		pv-spec:
			capacity:
			storage: 1Gi
			accessModes:
			- ReadWriteOnce
			storageClassName: manual
			hostPath:
			path: "/var/vineyard/spill"
		pvc-spec:
			storageClassName: manual
			accessModes:
			- ReadWriteOnce
			resources:
			requests:
				storage: 512Mi
		'

    # deploy the vineyardd with spill mechanism on persistent storage from json file
	# also you could use the yaml file
	cat pv-pvc.json | vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
		--vineyardd.spill.config spill-path \
		--vineyardd.spill.path /var/vineyard/spill \
		-`)
)

// deployVineyarddCmd deploys the vineyardd on kubernetes
var deployVineyarddCmd = &cobra.Command{
	Use:     "vineyardd",
	Short:   "Deploy the vineyardd on kubernetes",
	Long:    deployVineyarddLong,
	Example: deployVineyarddExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgsOrInput(cmd, args)
		// Check if the input is coming from stdin
		str, err := util.ReadJsonFromStdin(args)
		if err != nil {
			log.Fatal(err, "failed to parse from stdin")
		}
		if str != "" {
			flags.VineyardSpillPVandPVC = str
		}

		client := util.KubernetesClient()
		util.CreateNamespaceIfNotExist(client)

		vineyardd, err := BuildVineyard()
		if err != nil {
			log.Fatal(err, "failed to build vineyardd")
		}

		log.Info("wait for the vineyardd to be ready")
		var waitVineyarddFuc func(vineyardd *v1alpha1.Vineyardd) bool
		if flags.Wait {
			waitVineyarddFuc = func(vineyardd *v1alpha1.Vineyardd) bool {
				return vineyardd.Status.ReadyReplicas == int32(vineyardd.Spec.Replicas)
			}
		}
		if err := retry.Do(
			func() error {
				err := util.Create(client, vineyardd, waitVineyarddFuc)
				if err != nil {
					return errors.Wrap(err, "failed to create vineyardd")
				}
				return nil
			},
			retry.RetryIf(func(err error) bool {
				return strings.Contains(err.Error(), "connection refused") ||
					strings.Contains(err.Error(), "EOF")
			}),
			retry.Delay(5*time.Second),
		); err != nil {
			log.Fatal(err, "failed to retry applying vineyardd")
		}

		log.Info("Vineyardd is ready.")
	},
}

func BuildVineyard() (*v1alpha1.Vineyardd, error) {
	// Use file as first priority
	if flags.VineyarddFile != "" {
		vineyardd, err := BuildVineyardManifestFromFile()
		if err != nil {
			log.Fatal(err, "failed to build the vineyardd from file")
		}
		return vineyardd, nil

	}
	vineyardd, err := BuildVineyardManifestFromInput()
	if err != nil {
		log.Fatal(err, "failed to build the vineyardd from input")
	}
	// parse the volume and volume mounts
	if flags.VineyardVolume != "" {
		volumes, err := util.ParseVolume(flags.VineyardVolume)
		if err != nil {
			log.Fatal(err, "failed to parse the volumes")
		}
		vineyardd.Spec.Volumes = *volumes
	}
	if flags.VineyardVolumeMount != "" {
		volumeMounts, err := util.ParseVolumeMount(flags.VineyardVolumeMount)
		if err != nil {
			log.Fatal(err, "failed to parse the volume mounts")
		}
		vineyardd.Spec.VolumeMounts = *volumeMounts
	}
	if flags.VineyardSecurityContext != "" {
		securityContext, err := util.ParseSecurityContext(flags.VineyardSecurityContext)
		if err != nil {
			log.Fatal(err, "failed to parse the security context")
		}
		vineyardd.Spec.SecurityContext = *securityContext
	}
	return vineyardd, nil
}

func BuildVineyardManifestFromInput() (*v1alpha1.Vineyardd, error) {
	opts := &flags.VineyarddOpts
	envs := &flags.VineyardContainerEnvs

	if len(*envs) != 0 {
		vineyardContainerEnvs, err := util.ParseEnvs(*envs)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse envs")
		}
		opts.Vineyard.Env = append(opts.Vineyard.Env, vineyardContainerEnvs...)
	}

	spillPVandPVC := flags.VineyardSpillPVandPVC
	if spillPVandPVC != "" {
		pv, pvc, err := util.GetPVAndPVC(spillPVandPVC)
		if err != nil {
			return nil, errors.Wrap(err, "failed to get the pv and pvc of vineyard spill")
		}
		opts.Vineyard.Spill.PersistentVolumeSpec = *pv
		opts.Vineyard.Spill.PersistentVolumeClaimSpec = *pvc
	}

	vineyardd := &v1alpha1.Vineyardd{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.VineyarddName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: *opts,
	}
	return vineyardd, nil
}

// BuildVineyardManifestFromFile builds the vineyardd from file
func BuildVineyardManifestFromFile() (*v1alpha1.Vineyardd, error) {
	vineyardd := &v1alpha1.Vineyardd{}

	manifest, err := util.ReadFromFile(flags.VineyarddFile)
	if err != nil {
		return nil, err
	}
	decoder := util.Deserializer()
	gvk := vineyardd.GroupVersionKind()
	_, _, err = decoder.Decode([]byte(manifest), &gvk, vineyardd)
	if err != nil {
		return nil, err
	}
	if vineyardd.Namespace == "" {
		vineyardd.Namespace = flags.GetDefaultVineyardNamespace()
	}
	return vineyardd, err
}

func NewDeployVineyarddCmd() *cobra.Command {
	return deployVineyarddCmd
}

func init() {
	flags.ApplyVineyarddOpts(deployVineyarddCmd)
}
