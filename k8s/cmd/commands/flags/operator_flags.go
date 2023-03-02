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
	// the version of operator
	OperatorVersion string

	// the local path of operator kustomization directory
	KustomizeDir string
)

func ApplyOperatorOpts(cmd *cobra.Command) {
	cmd.Flags().
		StringVarP(&OperatorVersion, "version", "v", "dev",
			"the version of kustomize dir from github repo")
	cmd.Flags().StringVarP(&KustomizeDir, "local", "l", "",
		"the local kustomize dir")
}
