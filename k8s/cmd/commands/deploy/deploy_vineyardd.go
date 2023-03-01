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
	"log"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

// deployVineyarddCmd deploys the vineyardd on kubernetes
var deployVineyarddCmd = &cobra.Command{
	Use:   "vineyardd",
	Short: "Deploy the vineyardd on kubernetes",
	Long: `Deploy the vineyardd on kubernetes. You could deploy a customized vineyardd 
from stdin or file. 

For example:

# deploy the default vineyard on kubernetes
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd

# deploy the vineyardd with customized image
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
--image vineyardd:v0.12.2

# deploy the vineyardd with spill mechnism on persistent storage from json string
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
--vineyard.spill.config spill-path \
--vineyard.spill.path /var/vineyard/spill \
--vineyard.spill.pv-pvc-spec '{
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

# deploy the vineyardd with spill mechnism on persistent storage from yaml string
vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
--vineyard.spill.config spill-path \
--vineyard.spill.path /var/vineyard/spill \
--vineyard.spill.pv-pvc-spec \
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

# deploy the vineyardd with spill mechnism on persistent storage from json file
# also you could use the yaml file
cat pv-pvc.json | vineyardctl -n vineyard-system --kubeconfig /home/gsbot/.kube/config deploy vineyardd \
--vineyard.spill.config spill-path \
--vineyard.spill.path /var/vineyard/spill \
-`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) > 0 && args[0] != "-" {
			util.ErrLogger.Fatal("invalid argument: ", args)
		}

		// Check if the input is coming from stdin
		str, err := util.ReadJsonFromStdin(args)
		if err != nil {
			log.Fatalf("failed to parse from stdin: %v", err)
		}
		if str != "" {
			flags.VineyardSpillPVandPVC = str
		}
		client := util.KubernetesClient()

		vineyardd, err := BuildVineyardManifest()
		if err != nil {
			util.ErrLogger.Fatal("failed to build the vineyardd: ", err)
		}
		if err := waitVineyardReady(client, vineyardd); err != nil {
			util.ErrLogger.Fatal("failed to wait vineyardd ready: ", err)
		}

		util.InfoLogger.Println("Vineyardd is ready.")
	},
}

func BuildVineyardManifest() (*v1alpha1.Vineyardd, error) {
	opts := &flags.VineyarddOpts
	envs := &flags.VineyardContainerEnvs

	spillPVandPVC := flags.VineyardSpillPVandPVC
	if len(*envs) != 0 {
		vineyardContainerEnvs, err := util.ParseEnvs(*envs)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse envs")
		}
		opts.VineyardConfig.Env = append(opts.VineyardConfig.Env, vineyardContainerEnvs...)
	}

	if spillPVandPVC != "" {
		spillPVandPVCJson, err := util.ConvertToJson(spillPVandPVC)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pv of vineyard spill")
		}
		spillPVSpec, spillPVCSpec, err := util.ParsePVandPVCSpec(spillPVandPVCJson)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pvc of vineyard spill")
		}
		opts.VineyardConfig.SpillConfig.PersistentVolumeSpec = *spillPVSpec
		opts.VineyardConfig.SpillConfig.PersistentVolumeClaimSpec = *spillPVCSpec
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

func NewDeployVineyarddCmd() *cobra.Command {
	return deployVineyarddCmd
}

func init() {
	flags.ApplyVineyarddOpts(deployVineyarddCmd)
}

// wait for the vineyard cluster to be ready
func waitVineyardReady(c client.Client, vineyardd *v1alpha1.Vineyardd) error {
	return util.Wait(func() (bool, error) {
		err := c.Create(context.TODO(), vineyardd)
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return false, nil
		}
		return true, nil
	})
}
