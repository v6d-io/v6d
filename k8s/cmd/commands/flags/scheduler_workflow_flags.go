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

import "github.com/spf13/cobra"

var (
	// WorkflowFile is the path of workflow file
	WorkflowFile string

	// WithoutCRD is the flag to indicate whether the CRD is installed
	WithoutCRD bool
)

func ApplySchedulerWorkflowOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&WorkflowFile, "file", "f", "",
		"the path of workflow file")
	cmd.Flags().BoolVarP(&WithoutCRD, "without-crd", "", false,
		"whether the CRD(especially for GlobalObject and LocalObject) is installed")
}
