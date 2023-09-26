/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

package v1alpha1

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/avast/retry-go"
	"github.com/stretchr/testify/assert"
	"github.com/v6d-io/v6d/k8s/pkg/log"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	//+kubebuilder:scaffold:imports
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

// These tests use Ginkgo (BDD-style Go testing framework). Refer to
// http://onsi.github.io/ginkgo/ to learn more about Ginkgo.

var (
	cfg       *rest.Config
	k8sClient client.Client
	testEnv   *envtest.Environment
	ctx       context.Context
	cancel    context.CancelFunc
)

func Test_webhook(t *testing.T) {
	var GinkgoWriter io.Writer
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))

	ctx, cancel = context.WithCancel(context.TODO())

	log.Info("Bootstrapping test environment")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: false,
		WebhookInstallOptions: envtest.WebhookInstallOptions{
			Paths: []string{filepath.Join("..", "..", "..", "config", "webhook", "manifests.yaml")},
		},
	}

	var err error
	// cfg is defined in this file globally.
	cfg, err = testEnv.Start()
	assert.NoError(t, err)
	assert.NotNil(t, cfg)

	scheme := runtime.NewScheme()
	err = AddToScheme(scheme)
	assert.NoError(t, err)

	err = admissionv1beta1.AddToScheme(scheme)
	assert.NoError(t, err)

	//+kubebuilder:scaffold:scheme

	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme})
	assert.NoError(t, err)
	assert.NotNil(t, k8sClient)

	// start webhook server using Manager
	webhookInstallOptions := &testEnv.WebhookInstallOptions
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:             scheme,
		Host:               webhookInstallOptions.LocalServingHost,
		Port:               webhookInstallOptions.LocalServingPort,
		CertDir:            webhookInstallOptions.LocalServingCertDir,
		LeaderElection:     false,
		MetricsBindAddress: "0",
	})
	assert.NoError(t, err)

	err = (&Operation{}).SetupWebhookWithManager(mgr)
	assert.NoError(t, err)

	err = (&Sidecar{}).SetupWebhookWithManager(mgr)
	assert.NoError(t, err)

	err = (&Backup{}).SetupWebhookWithManager(mgr)
	assert.NoError(t, err)

	err = (&Recover{}).SetupWebhookWithManager(mgr)
	assert.NoError(t, err)

	err = (&CSIDriver{}).SetupWebhookWithManager(mgr)
	assert.NoError(t, err)

	//+kubebuilder:scaffold:webhook

	go func() {
		err = mgr.Start(ctx)
		assert.NoError(t, err)
	}()

	// wait for the webhook server to get ready
	dialer := &net.Dialer{Timeout: time.Second}
	addrPort := fmt.Sprintf(
		"%s:%d",
		webhookInstallOptions.LocalServingHost,
		webhookInstallOptions.LocalServingPort,
	)
	err = retry.Do(func() error {
		conn, err := tls.DialWithDialer(
			dialer,
			"tcp",
			addrPort,
			&tls.Config{InsecureSkipVerify: true},
		)
		if err != nil {
			return err
		}
		conn.Close()
		return nil
	})
	assert.NoError(t, err)

	cancel()
	time.Sleep(1 * time.Second)
	log.Info("tearing down the test environment")
	err = testEnv.Stop()
	assert.NoError(t, err)
}
