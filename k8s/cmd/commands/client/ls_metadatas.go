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
	lsMetadatasLong = util.LongDesc(`List vineyard metadatas and support IPC socket,
	RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
	every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or 
	VINEYARD_RPC_SOCKET.`)

	lsMetadatasExample = util.Examples(`
	# List no more than 10 vineyard metadatas
	vineyardctl ls metadatas --limit 10 --ipc-socket /var/run/vineyard.sock
	
	# List no more than 1000 vineyard metadatas
	vineyardctl ls metadatas --rpc-socket 127.0.0.1:9600 --limit 1000
	
	# List vineyard metadatas with the name matching the regex pattern
	vineyardctl ls metadatas --pattern "vineyard::Blob" --ipc-socket /var/run/vineyard.sock

	# List vineyard metadatas of the vineyard deployment
	vineyardctl ls metadatas --deployment-name vineyardd-sample -n vineyard-system --limit 1000
	
	# List vineyard metadatas sorted by the instance id
	vineyardctl ls metadatas --sorted-key instance_id --limit 1000 --ipc-socket /var/run/vineyard.sock

	# List vineyard metadatas sorted by the type and print the output as json format
	vineyardctl ls metadatas --sorted-key type --limit 1000 --format json --ipc-socket /var/run/vineyard.sock
	`)
)

// lsMetadatas is to list vineyard objects
var lsMetadatas = &cobra.Command{
	Use:     "metadatas",
	Short:   "List vineyard metadatas",
	Long:    lsMetadatasLong,
	Example: lsMetadatasExample,
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
		output.WithFilter(false).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)
		Output = output
	},
}

func NewLsMetadatasCmd() *cobra.Command {
	return lsMetadatas
}

func init() {
	flags.ApplyConnectOpts(lsMetadatas)
	flags.ApplyLsOpts(lsMetadatas)
	flags.ApplyOutputOpts(lsMetadatas)
}
