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
	GetObjectLong = util.LongDesc(`Get vineyard object and support IPC socket,
	RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
	every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or
	VINEYARD_RPC_SOCKET.`)

	GetObjectExample = util.Examples(`
	# Get vineyard object with the given vineyard object_id and the ipc socket
	vineyardctl get object --object_id xxxxxxxx --ipc-socket /var/run/vineyard.sock

	# Get vineyard object with the given vineyard object_id and the rpc socket
	vineyardctl get object --object_id xxxxxxxx --rpc-socket 127.0.0.1:9600
	`)
)

// lsObjects is to list vineyard objects
var GetObject = &cobra.Command{
	Use:     "object",
	Short:   "Get vineyard object",
	Long:    GetObjectLong,
	Example: GetObjectExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}
		metas, err := client.GetMetaData(flags.Object_id, flags.SyncRemote)
		if err != nil {
			log.Fatal(err, "failed to get vineyard object's metadata")
		}
		meta := metas.MetaData()
		//fmt.Println(meta)
		metadatas := make(map[string]map[string]any)
		metadatas[flags.Object_id] = meta
		output := util.NewOutput(&metadatas, nil, nil)

		// set the output options
		output.WithFilter(true).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)
		Output = output
	},
}

func NewGetObjectCmd() *cobra.Command {
	return GetObject
}

func init() {
	flags.ApplyConnectOpts(GetObject)
	flags.ApplyGetOpts(GetObject)
	flags.ApplyOutputOpts(GetObject)
}
