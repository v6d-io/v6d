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
	"log"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

// deleteCertManagerCmd deletes the vineyard operator on kubernetes
var deleteCertManagerCmd = &cobra.Command{
	Use:   "cert-manager",
	Short: "Delete the cert-manager on kubernetes",
	Long: `Delete the cert-manager in the cert-manager namespace. You should specify 
the version of deployed cert-manager and the default version is v1.9.1. 
For example:

# delete the default version(v1.9.1) of cert-manager
vineyardctl -k /home/gsbot/.kube/config delete cert-manager

# delete the specific version of cert-manager
vineyardctl -k /home/gsbot/.kube/config delete cert-manager -v 1.11.0`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("delete cert-manager", args); err != nil {
			log.Fatal("failed to validate delete cert-manager args and flags: ", err)
		}

		kubeClient, err := util.GetKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		certManagerManifests, err := util.GetCertManagerManifests(util.GetCertManagerURL())
		if err != nil {
			log.Fatal("failed to get cert-manager manifests: ", err)
		}

		if err := util.DeleteManifests(kubeClient, []byte(certManagerManifests), ""); err != nil {
			log.Fatal("failed to delete cert-manager manifests: ", err)
		}
		log.Println("Cert-Manager is deleted.")
	},
}

func NewDeleteCertManagerCmd() *cobra.Command {
	return deleteCertManagerCmd
}

func init() {
	flags.NewCertManagerOpts(deleteCertManagerCmd)
}
