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
	"fmt"

	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployCSIDriverLong = util.LongDesc(`
	Deploy the Vineyard CSI Driver on kubernetes. 
	The CR is a cluster-scoped resource, and can only be created once.`)

	deployCSIDriverExample = util.Examples(`
	# deploy the Vineyard CSI Driver named vineyard-csi-sample on kubernetes
	# Notice, the clusters are built as {vineyard-deployment-namespace}/{vineyard-deployment-name}
	# and sperated by comma, e.g. vineyard-system/vineyardd-sample, default/vineyardd-sample
	# They must be created before deploying the Vineyard CSI Driver.
	vineyardctl deploy csidriver --name vineyard-csi-sample \
		--clusters vineyard-system/vineyardd-sample,default/vineyardd-sample`)
)

// deployCSIDriverCmd deploys the vineyard csi driver on kubernetes
var deployCSIDriverCmd = &cobra.Command{
	Use:     "csidriver",
	Short:   "Deploy the vineyard csi driver on kubernetes",
	Long:    deployCSIDriverLong,
	Example: deployCSIDriverExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgsOrInput(cmd, args)

		client := util.KubernetesClient()
		log.Info("building CSIDriver cr")
		csiDriver, err := BuildCSIDriver()
		if err != nil {
			log.Fatal(err, "Failed to build csi driver")
		}
		log.Info("creating csi driver")
		if err := util.Create(client, csiDriver, func(csiDriver *v1alpha1.CSIDriver) bool {
			return csiDriver.Status.State == v1alpha1.CSIDriverRunning
		}); err != nil {
			log.Fatal(err, "failed to create/wait Vineyard CSI Driver")
		}

		log.Info("Vineyard CSI Driver is ready.")
	},
}

func checkVolumeBindingMode(mode string) bool {
	switch mode {
	case "WaitForFirstConsumer", "Immediate":
		return true
	default:
		return false
	}
}

func BuildCSIDriver() (*v1alpha1.CSIDriver, error) {
	clusters, err := util.ParseVineyardClusters(flags.VineyardClusters)
	if err != nil {
		return nil, err
	}
	csiDriver := &v1alpha1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.CSIDriverName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: flags.CSIDriverOpts,
	}
	if !checkVolumeBindingMode(csiDriver.Spec.VolumeBindingMode) {
		return nil, fmt.Errorf("invalid volume binding mode: %s, "+
			"only support WaitForFirstConsumer and Immediate",
			csiDriver.Spec.VolumeBindingMode)
	}
	csiDriver.Spec.Clusters = *clusters
	csiDriver.Spec.EnableVerboseLog = flags.Verbose
	return csiDriver, nil
}

func NewDeployCSIDriverCmd() *cobra.Command {
	return deployCSIDriverCmd
}

func init() {
	flags.ApplyCSIDriverOpts(deployCSIDriverCmd)
}
