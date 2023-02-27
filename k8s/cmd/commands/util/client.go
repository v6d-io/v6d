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
	defaultscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	apiv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"sigs.k8s.io/controller-runtime/pkg/client"

	certmanagerv1 "github.com/cert-manager/cert-manager/pkg/apis/certmanager/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

var scheme = runtime.NewScheme()

func init() {
	_ = defaultscheme.AddToScheme(scheme)
	_ = v1alpha1.AddToScheme(scheme)
	_ = apiv1.AddToScheme(scheme)
	_ = certmanagerv1.AddToScheme(scheme)
}

func Scheme() *runtime.Scheme {
	return scheme
}

func Deserializer() runtime.Decoder {
	return serializer.NewCodecFactory(scheme).UniversalDeserializer()
}

// KubernetesClient return the kubernetes client
func KubernetesClient() client.Client {
	cfg, err := clientcmd.BuildConfigFromFlags("", flags.KubeConfig)
	if err != nil {
		cfg, err = rest.InClusterConfig()
		if err != nil {
			ErrLogger.Fatalf("Failed to resolve the KUBECONFIG: %+v", err)
		}
	}
	client, _ := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		ErrLogger.Fatalf("failed to create the kubernetes API client: %+v", err)
	}
	return client
}
