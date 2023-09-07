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
	"fmt"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	putLong = util.LongDesc(`Put basic data type into vineyard and only support IPC socket.
	It receives the flag --value as string and will print object id if succeed.
	If you don't specify the ipc socket every time, you can set it as the 
	environment variable VINEYARD_IPC_SOCKET.`)

	putExample = util.Examples(`
	# put value into vineyard with the given ipc socket
	vineyardctl put --value 12345 --ipc-socket /var/run/vineyard.sock 
	vineyardctl put --value hello,world --ipc-socket /var/run/vineyard.sock
	
	# If you set the environment variable VINEYARD_IPC_SOCKET
	# you can use the following command to get vineyard blob
	vineyardctl put --value 12345`)
)

// putCmd is to put basic data type into vineyard
var putCmd = &cobra.Command{
	Use:     "put",
	Short:   "Put basic data type into vineyard",
	Long:    putLong,
	Example: putExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}

		// Convert flags.Value (type string) to type [] byte
		value := []byte(flags.Value)
		object_id, err := client.PutBlob(value, uint64(len(flags.Value)))
		if err != nil {
			log.Errorf(err, fmt.Sprintf("failed to put value: %v", value))
		}
		log.Output(types.ObjectIDToString(object_id))
	},
}

func NewPutCmd() *cobra.Command {
	return putCmd
}

func init() {
	flags.ApplyConnectOpts(putCmd)
	flags.ApplyPutOpts(putCmd)
	flags.ApplyOutputOpts(putCmd)
}
