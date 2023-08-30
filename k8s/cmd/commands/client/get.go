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
package client

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var (
	getExample = util.Examples(`
	# Connect the vineyardd deployment with IPC client
	# Get the cluster info and output as table
	vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system`)
)

// getCmd is to get vineyard objects, metadatas, blobs or cluster-info
var getCmd = &cobra.Command{
	Use:     "get",
	Short:   "Get vineyard object, metadata, blob or cluster-info",
	Example: getExample,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		// disable stdout
		os.Stdout, _ = os.Open(os.DevNull)
	},
	PersistentPostRun: func(cmd *cobra.Command, args []string) {
		// enable stdout
		os.Stdout = Stdout
		Output.Print()
	},
}

func NewGetCmd() *cobra.Command {
	return getCmd
}

func init() {
	getCmd.AddCommand(NewGetClusterInfoCmd())
	getCmd.AddCommand(NewGetMetadataCmd())
	getCmd.AddCommand(NewGetBlobCmd())
	getCmd.AddCommand(NewGetObjectCmd())
}
