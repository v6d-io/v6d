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
	"log"
	"os"

	"github.com/spf13/cobra"
)

// get the default kubeconfig path
func getDefaultKubeconfig() string {
	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		kubeconfig = os.Getenv("HOME") + "/.kube/config"
	}
	return kubeconfig
}

var VineyardSystemNamespace = "vineyard-system"

// get the default vineyard namespace
func GetDefaultVineyardNamespace() string {
	// we don't use the default namespace for vineyard
	if Namespace == "default" {
		return VineyardSystemNamespace
	}
	return Namespace
}

var defaultKubeconfig = getDefaultKubeconfig()

// kubeconfig path
var Kubeconfig string

// Namespace for operation
var Namespace string

var rootCmd = &cobra.Command{
	Use:   "vineyardctl [command]",
	Short: "vineyardctl is the command-line tool for working with the Vineyard Operator",
	Long: `vineyardctl is the command-line tool for working with the Vineyard Operator. 
It supports creating, deleting and checking status of Vineyard Operator. It also 
supports managing the vineyard relevant components such as vineyardd and pluggable 
drivers`,

	Run: func(cmd *cobra.Command, args []string) {
		log.Println("Welcome to vineyardctl")
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatal("failed to execute root command: ", err)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVarP(&Kubeconfig, "kubeconfig", "k", defaultKubeconfig, "kubeconfig path for the kubernetes cluster")
	rootCmd.PersistentFlags().StringVarP(&Namespace, "namespace", "n", "default", "the namespace for operation")
	rootCmd.AddCommand(NewDeployCmd())
	rootCmd.AddCommand(NewCreateCmd())
	rootCmd.AddCommand(NewDeleteCmd())
	//disable completion command
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}
