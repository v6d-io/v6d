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
	getBlobLong = util.LongDesc(`Get vineyard blob and only support IPC socket.
	If you don't specify the ipc socket every time, you can set it as the 
	environment variable VINEYARD_IPC_SOCKET.`)

	getBlobExample = util.Examples(`
	# Get vineyard blob with the given vineyard object_id and the ipc socket
	vineyardctl get blob --object_id xxxxxxxx --ipc-socket /var/run/vineyard.sock 

	# Get vineyard blob with the given vineyard object_id and the ipc socket
	# and set the unsafe to be true
	vineyardctl get blob --object_id xxxxxxxx --unsafe --ipc-socket /var/run/vineyard.sock 
	
	# If you set the environment variable VINEYARD_IPC_SOCKET
	# you can use the following command to get vineyard blob
	vineyardctl get blob --object_id xxxxxxxx`)
)

// getBLobs is to get vineyard blobs
var getBlob = &cobra.Command{
	Use:     "blob",
	Short:   "Get vineyard blob",
	Long:    getBlobLong,
	Example: getBlobExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}

		// get the blob
		blob, err := client.GetBlob(flags.Object_id, flags.Unsafe)
		if err != nil {
			log.Fatal(err, "failed to get vineyard blob")
		}
		output := util.NewOutput(nil, &blob, nil)
		// set the output options
		output.WithFilter(false).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)
		Output = output
	},
}

func NewGetBlobCmd() *cobra.Command {
	return getBlob
}

func init() {
	flags.ApplyConnectOpts(getBlob)
	flags.ApplyGetOpts(getBlob)
	flags.ApplyOutputOpts(getBlob)
}
