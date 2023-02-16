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
package util

import (
	cmapi "github.com/cert-manager/cert-manager/pkg/apis/certmanager/v1"
	vineyardV1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	apiv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoScheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var (
	CmdScheme = runtime.NewScheme()
)

// add k8s apis scheme, vineyard v1alpha1 scheme and cert-manager v1alpha1 scheme
func init() {
	_ = clientgoScheme.AddToScheme(CmdScheme)
	_ = vineyardV1alpha1.AddToScheme(CmdScheme)
	_ = apiv1.AddToScheme(CmdScheme)
	_ = cmapi.AddToScheme(CmdScheme)
}

// GetKubeClient return the kubernetes client
func GetKubeClient() (client.Client, error) {
	cfg, err := clientcmd.BuildConfigFromFlags("", flags.Kubeconfig)
	if err != nil {
		return nil, err
	}
	client, _ := client.New(cfg, client.Options{Scheme: CmdScheme})
	if err != nil {
		return nil, err
	}

	return client, nil
}
