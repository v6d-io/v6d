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
package create

import (
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	createOperationLong = util.LongDesc(`
	Insert an operation in a workflow based on vineyard cluster.
	You could create a assembly or repartition operation in a
	workflow. Usually, the operation should be created between
	the workloads: job1 -> operation -> job2.`)

	createOperationExample = util.Examples(`
	# create a local assembly operation between job1 and job2
	vineyardctl create operation --name assembly \
		--type local \
		--require job1 \
		--target job2 \
		--timeoutSeconds 600

	# create a distributed assembly operation between job1 and job2
	vineyardctl create operation --name assembly \
		--type distributed \
		--require job1 \
		--target job2 \
		--timeoutSeconds 600

	# create a dask repartition operation between job1 and job2
	vineyardctl create operation --name repartition \
		--type dask \
		--require job1 \
		--target job2 \
		--timeoutSeconds 600`)
)

// createOperationCmd creates the specific operation in a workflow.
var createOperationCmd = &cobra.Command{
	Use:     "operation",
	Short:   "Insert an operation in a workflow based on vineyard cluster",
	Long:    createOperationLong,
	Example: createOperationExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		client := util.KubernetesClient()
		util.CreateNamespaceIfNotExist(client)
		operation := buildOperation()

		log.Info("creating Operation cr")
		if err := util.Create(client, operation, func(operation *v1alpha1.Operation) bool {
			return operation.Status.State != v1alpha1.OperationSucceeded
		}); err != nil {
			log.Fatal(err, "failed to create/wait operation")
		}

		log.Info("operation executed successfully")
	},
}

func buildOperation() *v1alpha1.Operation {
	namespace := flags.GetDefaultVineyardNamespace()
	operation := &v1alpha1.Operation{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.OperationName,
			Namespace: namespace,
		},
		Spec: flags.OperationOpts,
	}
	return operation
}

func NewCreateOperationCmd() *cobra.Command {
	return createOperationCmd
}

func init() {
	flags.ApplyOperationOpts(createOperationCmd)
}
