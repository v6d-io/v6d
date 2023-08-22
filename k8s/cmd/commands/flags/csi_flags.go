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
	// Endpoint is the endpoint of vineyard csi driver
	Endpoint string

	// NodeID is the node id of vineyard csi driver
	NodeID string

	// StateFilePath is the path of state file
	StateFilePath string
)

func ApplyCsiOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&Endpoint, "endpoint", "f", "",
		"the endpoint of vineyard csi driver")
	cmd.Flags().StringVarP(&NodeID, "nodeid", "", "",
		"the node id of vineyard csi driver")
	cmd.Flags().StringVarP(&StateFilePath, "state-file-path", "", "/csi/state",
		"the path of state file")
}
