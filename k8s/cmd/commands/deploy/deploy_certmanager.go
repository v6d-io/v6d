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

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	certmanagerapiv1 "github.com/cert-manager/cert-manager/pkg/apis/certmanager/v1"
	certmanagermetav1 "github.com/cert-manager/cert-manager/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployCertManagerLong = util.LongDesc(`
	Deploy the cert-manager in the cert-manager namespace. You could
	specify a stable or development version of the cert-manager and
	we suppose not to create a new namespace to install the
	cert-manager. The default version is v1.9.1.`)

	deployCertManagerExample = util.Examples(`
	# install the default version(v1.9.1) in the cert-manager namespace
	# wait for the cert-manager to be ready(default option)
	vineyardctl --kubeconfig $HOME/.kube/config deploy cert-manager

	# install the default version(v1.9.1) in the cert-manager namespace
	# not to wait for the cert-manager to be ready, but we does not recommend
	# to do this, because there may be errors caused by the cert-manager
	# not ready
	vineyardctl --kubeconfig $HOME/.kube/config deploy cert-manager \
		--wait=false

	# install the specific version of cert-manager
	vineyardctl --kubeconfig $HOME/.kube/config deploy cert-manager -v 1.11.0`)
)

// deployCertManagerCmd deploys the vineyard operator on kubernetes
var deployCertManagerCmd = &cobra.Command{
	Use:     "cert-manager",
	Short:   "Deploy the cert-manager on kubernetes",
	Long:    deployCertManagerLong,
	Example: deployCertManagerExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		certManagerManifests, err := util.GetCertManagerManifests(util.GetCertManagerURL())
		if err != nil {
			log.Fatal(err, "failed to get cert-manager manifests")
		}

		if err := util.ApplyManifests(client, certManagerManifests,
			""); err != nil {
			log.Fatal(err, "failed to apply cert-manager manifests")
		}

		if flags.Wait {
			if err := waitCertManagerReady(client); err != nil {
				log.Fatal(err, "failed to wait cert-manager ready")
			}
		}

		log.Info("Cert-Manager is ready.")
	},
}

func NewDeployCertManagerCmd() *cobra.Command {
	return deployCertManagerCmd
}

func init() {
	flags.ApplyCertManagerOpts(deployCertManagerCmd)
}

// wait cert-manager to be ready
func waitCertManagerReady(c client.Client) error {
	return util.Wait(func() (bool, error) {
		// create a dummy selfsigned issuer
		dummyIssue := &certmanagerapiv1.Issuer{
			ObjectMeta: metav1.ObjectMeta{
				Name:      string("selfsigned-issuer"),
				Namespace: "default",
			},
			Spec: certmanagerapiv1.IssuerSpec{
				IssuerConfig: certmanagerapiv1.IssuerConfig{
					SelfSigned: &certmanagerapiv1.SelfSignedIssuer{},
				},
			},
		}
		// create a dummy selfsigned certificate
		dummyCert := &certmanagerapiv1.Certificate{
			ObjectMeta: metav1.ObjectMeta{
				Name:      string("selfsigned-cert"),
				Namespace: "default",
			},
			Spec: certmanagerapiv1.CertificateSpec{
				DNSNames:   []string{"example.com"},
				SecretName: "selfsigned-cert-tls",
				IssuerRef: certmanagermetav1.ObjectReference{
					Name: "selfsigned-issuer",
				},
			},
		}

		err := c.Create(context.TODO(), dummyIssue)
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return false, nil
		}

		err = c.Create(context.TODO(), dummyCert)
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return false, nil
		}

		_ = c.Delete(context.TODO(), dummyIssue)
		_ = c.Delete(context.TODO(), dummyCert)

		return true, nil
	})
}
