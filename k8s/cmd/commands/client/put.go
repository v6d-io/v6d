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
	"log"

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

		// Convert flags.Value (type string) to type [] byte
		value := []byte(flags.Value)
		object_id, err := client.PutBlob(value, uint64(len(flags.Value)))
		if err != nil {
			log.Printf("failed to put value: %s", err)
		}
		objectId := fmt.Sprintf("o%016x", object_id)
		fmt.Println("object id : ", objectId)
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
