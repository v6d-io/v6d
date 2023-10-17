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
)

var (
	// ArgoWorkflowFile is the name of argo workflow file
	ArgoWorkflowFile string

	// TemplateName is the name of workflow template which will be injected
	// with vineyard volumes
	WorkflowTemplates []string

	// MountPath is the mount path of vineyard volumes
	MountPath string

	// VineyardCluster is the name of vineyard cluster which the argo workflow
	// will use
	VineyardCluster string

	// Dag is the name of dag which will be injected with vineyard volumes
	Dag string

	// Tasks contains the set of task names under the dag
	Tasks []string

	// OutputAsFile means whether to output the injected workflow as a file
	OutputAsFile bool
)

func ApplyInjectArgoWorkflowOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&ArgoWorkflowFile, "file", "f", "",
		"The file name of argo workflow")
	cmd.Flags().StringSliceVarP(&WorkflowTemplates, "templates", "t",
		[]string{}, "The name of workflow template which will be injected with vineyard volumes")
	cmd.Flags().StringVarP(&MountPath, "mount-path", "", "",
		"The mount path of vineyard volumes")
	cmd.Flags().StringVarP(&VineyardCluster, "vineyard-cluster", "", "",
		"The name of vineyard cluster which the argo workflow will use")
	cmd.Flags().StringVarP(&Dag, "dag", "", "",
		"The name of dag which will be injected with vineyard volumes")
	cmd.Flags().StringSliceVarP(&Tasks, "tasks", "", []string{},
		"The set of task names under the dag")
	cmd.Flags().BoolVarP(&OutputAsFile, "output-as-file", "", false,
		"Whether to output the injected workflow as a file, default is false"+
			"The output file name will add a suffix '_with_vineyard' to the original file name")
}
