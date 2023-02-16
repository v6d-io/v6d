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
	"log"
	"time"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// deployVineyarddCmd deploys the vineyardd on kubernetes
var deployVineyarddCmd = &cobra.Command{
	Use:   "vineyardd",
	Short: "Deploy the vineyardd on kubernetes",
	Long: `Deploy the vineyardd on kubernetes. You could deploy a customized vineyardd 
from stdin or file. 

For example:

# deploy the default vineyard on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config deploy vineyardd

# deploy the vineyardd with customized image
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config deploy vineyardd --image vineyardd:v0.12.2

# deploy the customized vineyardd from file on kubernetes
vineyardctl -n vineyard-system -k /home/gsbot/.kube/config deploy vineyardd --file vineyardd.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("deploy vineyardd", args); err != nil {
			log.Fatal("failed to validate deploy vineyardd command args and flags: ", err)
		}
		kubeClient, err := util.GetKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		vineyardd, err := buildVineyardManifest()
		if err != nil {
			log.Fatal("failed to build the vineyardd: ", err)
		}
		if err := waitVineyardReady(kubeClient, vineyardd); err != nil {
			log.Fatal("failed to wait vineyardd ready: ", err)
		}

		log.Println("Vineyardd is ready.")
	},
}

func buildVineyardManifest() (*v1alpha1.Vineyardd, error) {
	opts := &flags.VineyarddOpts
	envs := &flags.VineyardContainerEnvs
	spillPV := flags.VineyardSpillPVSpec
	spillPVC := flags.VineyardSpillPVCSpec
	if len(*envs) != 0 {
		vineyardContainerEnvs, err := util.ParseEnvs(*envs)
		if err != nil {
			return nil, fmt.Errorf("failed to parse envs: %v", err)
		}
		opts.VineyardConfig.Env = append(opts.VineyardConfig.Env, vineyardContainerEnvs...)
	}

	if spillPV != "" {
		vineyarddSpillPV, err := util.ParsePVSpec(spillPV)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the pv of vineyard spill: %v", err)
		}
		opts.VineyardConfig.SpillConfig.PersistentVolumeSpec = *vineyarddSpillPV
	}

	if spillPVC != "" {
		vineyardSpillPVC, err := util.ParsePVCSpec(spillPVC)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the pvc of vineyard spill: %v", err)
		}
		opts.VineyardConfig.SpillConfig.PersistentVolumeClaimSpec = *vineyardSpillPVC
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

// wait for the vineyard cluster to be ready
func waitVineyardReady(c client.Client, vineyardd *v1alpha1.Vineyardd) error {
	return wait.PollImmediate(1*time.Second, 300*time.Second, func() (bool, error) {
		err := c.Create(context.TODO(), vineyardd)
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return false, nil
		}
		return true, nil
	})
}

func NewDeployVineyarddCmd() *cobra.Command {
	return deployVineyarddCmd
}

func init() {
	flags.NewVineyarddOpts(deployVineyarddCmd)
}
