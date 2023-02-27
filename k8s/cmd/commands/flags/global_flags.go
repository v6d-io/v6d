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
package flags

import (
	"os"

	"github.com/spf13/cobra"
)

// defaultKubeConfig return the default kubeconfig path
func defaultKubeConfig() string {
	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		kubeconfig = os.Getenv("HOME") + "/.kube/config"
	}
	return kubeconfig
}

var defaultNamespace = "vineyard-system"

// GetDefaultVineyardNamespace return the default vineyard namespace
func GetDefaultVineyardNamespace() string {
	// we don't use the default namespace for vineyard
	if Namespace == "default" {
		return defaultNamespace
	}
	return Namespace
}

// kubeconfig path
var KubeConfig string

// Namespace for operation
var Namespace string

func ApplyGlobalFlags(cmd *cobra.Command) {
	cmd.PersistentFlags().
		StringVarP(&KubeConfig, "kubeconfig", "", defaultKubeConfig(), "kubeconfig path for the kubernetes cluster")
	cmd.PersistentFlags().
		StringVarP(&Namespace, "namespace", "n", defaultNamespace, "the namespace for operation")
}
