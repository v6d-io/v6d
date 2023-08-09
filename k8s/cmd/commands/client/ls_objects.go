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
	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	lsObjectsLong = util.LongDesc(`List vineyard objects and support IPC socket,
	RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
	every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or
	VINEYARD_RPC_SOCKET.`)

	lsObjectsExample = util.Examples(`
	# List no more than 10 vineyard objects
	vineyardctl ls objects --limit 10 --ipc-socket /var/run/vineyard.sock

	# List any vineyard objects and no more than 1000 objects
	vineyardctl ls objects --pattern "*" --ipc-socket /var/run/vineyard.sock --limit 1000

	# List vineyard objects with the name matching the regex pattern
	vineyardctl ls objects --pattern "vineyard::Tensor<.*>" --regex --ipc-socket /var/run/vineyard.sock

	# List vineyard objects and output as json format
	vineyardctl ls objects --format json --ipc-socket /var/run/vineyard.sock

	# List vineyard objects sorted by the typename
	vineyardctl ls objects --sorted-key typename --limit 1000 --ipc-socket /var/run/vineyard.sock`)
)

// lsObjects is to list vineyard objects
var lsObjects = &cobra.Command{
	Use:     "objects",
	Short:   "List vineyard objects",
	Long:    lsObjectsLong,
	Example: lsObjectsExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}
		metas, err := client.ListMetadatas(flags.Pattern, flags.Regex, flags.Limit)
		if err != nil {
			log.Fatal(err, "failed to list vineyard objects")
		}
		output := util.NewOutput(&metas, nil, nil)
		// set the output options
		output.WithFilter(true).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)
		Output = output
	},
}

func NewLsObjectsCmd() *cobra.Command {
	return lsObjects
}

func init() {
	flags.ApplyConnectOpts(lsObjects)
	flags.ApplyLsOpts(lsObjects)
	flags.ApplyOutputOpts(lsObjects)
}
