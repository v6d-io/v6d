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

var lsExample = util.Examples(`
	# Connect the vineyardd server with IPC client
	# List the vineyard objects no more than 10
	vineyardctl ls objects --limit 10 --ipc-socket /var/run/vineyard.sock

	# List the vineyard blobs no more than 10
	vineyardctl ls blobs --limit 10 --ipc-socket /var/run/vineyard.sock

	# List the vineyard objects with the specified pattern
	vineyardctl ls objects --pattern "vineyard::Tensor<.*>" --regex --ipc-socket /var/run/vineyard.sock

	# Connect the vineyardd server with RPC client
	# List the vineyard metadatas no more than 1000
	vineyardctl ls metadatas --rpc-socket 127.0.0.1:9600 --limit 1000

	# Connect the vineyard deployment with PRC client
	# List the vineyard objects no more than 1000
	vineyardctl ls objects --deployment-name vineyardd-sample -n vineyard-system`)

var (
	Stdout = os.Stdout
	Output *util.Output
)

// lsCmd is to list vineyard objects
var lsCmd = &cobra.Command{
	Use:     "ls",
	Short:   "List vineyard objects, metadatas or blobs",
	Example: lsExample,
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

func NewLsCmd() *cobra.Command {
	return lsCmd
}

func init() {
	lsCmd.AddCommand(NewLsMetadatasCmd())
	lsCmd.AddCommand(NewLsObjectsCmd())
	lsCmd.AddCommand(NewLsBlobsCmd())
}
