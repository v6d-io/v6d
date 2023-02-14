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
package commands

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	cmapi "github.com/cert-manager/cert-manager/pkg/apis/certmanager/v1"
	cmmeta "github.com/cert-manager/cert-manager/pkg/apis/meta/v1"
	"github.com/spf13/cobra"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// the default cert-manager manifest url
var DefaultCertManagerManifestURL = "https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml"

// CertManagerVersion is the version of cert-manager
var CertManagerVersion string

// deployCertManagerCmd deploys the vineyard operator on kubernetes
var deployCertManagerCmd = &cobra.Command{
	Use:   "cert-manager",
	Short: "Deploy the cert-manager on kubernetes",
	Long: `Deploy the cert-manager in the cert-manager namespace. You could specify 
a stable or development version of the cert-manager and we suppose not to create a
new namespace to install the cert-manager. The default version is v1.9.1. 
For example:

# install the default version(v1.9.1) in the cert-manager namespace
vineyardctl -k /home/gsbot/.kube/config deploy cert-manager

# install the specific version of cert-manager
vineyardctl -k /home/gsbot/.kube/config deploy cert-manager -v 1.11.0`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := ValidateNoArgs("deploy cert-manager", args); err != nil {
			log.Fatal("failed to validate deploy cert-manager args and flags: ", err)
		}
		kubeClient, err := getKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		certManagerManifests, err := getCertManagerManifests(getCertManagerURL())
		if err != nil {
			log.Fatal("failed to get cert-manager manifests: ", err)
		}

		if err := applyManifests(kubeClient, []byte(certManagerManifests), ""); err != nil {
			log.Fatal("failed to apply cert-manager manifests: ", err)
		}

		if err := waitCertManagerReady(kubeClient); err != nil {
			log.Fatal("failed to wait cert-manager ready: ", err)
		}

		log.Println("Cert-Manager is ready.")
	},
}

// wait cert-manager to be ready
func waitCertManagerReady(c client.Client) error {
	return wait.PollImmediate(10*time.Second, 300*time.Second, func() (bool, error) {
		// create a dummy selfsigned issuer
		dummyIssue := &cmapi.Issuer{
			ObjectMeta: metav1.ObjectMeta{
				Name:      string("selfsigned-issuer"),
				Namespace: "default",
			},
			Spec: cmapi.IssuerSpec{
				IssuerConfig: cmapi.IssuerConfig{
					SelfSigned: &cmapi.SelfSignedIssuer{},
				},
			},
		}
		// create a dummy selfsigned certificate
		dummyCert := &cmapi.Certificate{
			ObjectMeta: metav1.ObjectMeta{
				Name:      string("selfsigned-cert"),
				Namespace: "default",
			},
			Spec: cmapi.CertificateSpec{
				DNSNames:   []string{"example.com"},
				SecretName: "selfsigned-cert-tls",
				IssuerRef: cmmeta.ObjectReference{
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

func validateCertManager(agrs []string) error {
	if len(agrs) != 0 {
		return fmt.Errorf("the deploy cert-manager command doesn't need any args")
	}
	return nil
}

func getCertManagerURL() string {
	if CertManagerVersion != "1.9.1" {
		return strings.Replace(DefaultCertManagerManifestURL, "v1.9.1", "v"+CertManagerVersion, 1)
	}
	return DefaultCertManagerManifestURL
}

func getCertManagerManifests(certManagerManifestURL string) (string, error) {
	resp, err := http.Get(certManagerManifestURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	manifests, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(manifests), nil
}

func NewDeployCertManagerCmd() *cobra.Command {
	return deployCertManagerCmd
}

func init() {
	deployCertManagerCmd.Flags().StringVarP(&CertManagerVersion, "version", "v", "1.9.1", "the version of cert-manager")
}
