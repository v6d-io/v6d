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
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
)

var (
	// OperationOpts holds all configuration of operation Spec
	OperationOpts v1alpha1.OperationSpec

	// OperationName is the name of operation
	OperationName string
)

func ApplyOperationName(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&OperationName, "name", "", "", "the name of operation")
}

func ApplyOperationOpts(cmd *cobra.Command) {
	ApplyOperationName(cmd)
	cmd.Flags().StringVarP(&OperationOpts.Name, "kind", "", "",
		`the kind of operation, including "assembly" and "repartition"`)
	cmd.Flags().StringVarP(&OperationOpts.Type, "type", "", "",
		`the type of operation: for assembly, it can be "local" or "distributed"; `+
			`for repartition, it can be "dask"`)
	cmd.Flags().StringVarP(&OperationOpts.Require, "require", "", "",
		"the job that need an operation to be executed")
	cmd.Flags().StringVarP(&OperationOpts.Target, "target", "", "",
		"the job that need to be executed before this operation")
	cmd.Flags().Int64VarP(&OperationOpts.TimeoutSeconds, "timeoutSeconds", "", 600,
		"the timeout seconds of operation")
}
