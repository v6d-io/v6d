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
	lsBlobsLong = util.LongDesc(`List vineyard blobs and only support IPC socket.
	If you don't specify the ipc socket every time, you can set it as the 
	environment variable VINEYARD_IPC_SOCKET.`)

	lsBlobsExample = util.Examples(`
	# List no more than 10 vineyard blobs
	vineyardctl ls blobs --limit 10 --ipc-socket /var/run/vineyard.sock 

	# List no more than 1000 vineyard blobs
	vineyardctl ls blobs --ipc-socket /var/run/vineyard.sock --limit 1000
	
	# List vineyard blobs with the name matching
	vineyardctl ls blobs --pattern "vineyard::Tensor<.*>" --regex --ipc-socket /var/run/vineyard.sock
	
	# List vineyard blobs with the regex pattern
	vineyardctl ls blobs --pattern "*DataFrame*" --ipc-socket /var/run/vineyard.sock
	
	# If you set the environment variable VINEYARD_IPC_SOCKET
	# you can use the following command to list vineyard blobs
	vineyardctl ls blobs --limit 1000`)
)

// lsBlobs is to list vineyard blobs
var lsBlobs = &cobra.Command{
	Use:     "blobs",
	Short:   "List vineyard blobs",
	Long:    lsBlobsLong,
	Example: lsBlobsExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}

		blobs, err := client.ListBlobs(flags.Limit)
		if err != nil {
			log.Fatal(err, "failed to list vineyard blobs")
		}
		output := util.NewOutput(nil, &blobs, nil)
		// set the output options
		output.WithFilter(false).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)
		Output = output
	},
}

func NewLsBlobsCmd() *cobra.Command {
	return lsBlobs
}

func init() {
	flags.ApplyConnectOpts(lsBlobs)
	flags.ApplyLimitOpt(lsBlobs)
	flags.ApplyOutputOpts(lsBlobs)
}
