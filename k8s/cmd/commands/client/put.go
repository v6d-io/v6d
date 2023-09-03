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
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var (
	putExample = util.Examples(`
	# Connect the vineyardd deployment with IPC client
	# Get the cluster info and output as table
	vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system`)
)

// getCmd is to get vineyard objects, metadatas, blobs or cluster-info
var putCmd = &cobra.Command{
	Use:     "put",
	Short:   "Put vineyard object, metadata, blob or cluster-info",
	Example: putExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}

		flags.Value = []uint{1}
		// Convert flags.Value (type [] uint) to type [] byte
		value := make([]byte, len(flags.Value))
		for i, v := range flags.Value {
			value[i] = byte(v)
		}
		//object_id, _ := client.PutBlob(value, uint64(len(flags.Value)))
		object_id, _ := client.PutBlob(value, 64)
		a := fmt.Sprintf("o%016x", object_id)
		fmt.Println(a)
		/*output := util.NewOutput(nil, &blob, nil)
		// set the output options
		output.WithFilter(false).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)
		Output = output*/
	},
}

func NewPutCmd() *cobra.Command {
	return putCmd
}

func init() {
	flags.ApplyConnectOpts(putCmd)
	flags.ApplyGetOpts(putCmd)
	flags.ApplyOutputOpts(putCmd)
}
