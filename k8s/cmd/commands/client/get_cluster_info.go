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
	getClusterInfoLong = util.LongDesc(`Get vineyard cluster info, including
	the instanceId, hostName, node name and so on.`)

	getClusterInfoExample = util.Examples(`
	# Get the cluster info of vineyard deployment and output as table
	vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system

	# Get the cluster info of vineyard deployment and output as json
	vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system -o json
	
	# Get the cluster info via IPC socket
	vineyardctl get cluster-info --ipc-socket /var/run/vineyard.sock`)
)

// getClusterInfo is to get vineyard cluster info
var getClusterInfo = &cobra.Command{
	Use:     "cluster-info",
	Short:   "Get vineyard cluster info",
	Long:    getClusterInfoLong,
	Example: getClusterInfoExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		// check the client socket
		util.CheckClientSocket(cmd, args)

		client, ch := util.NewClient()
		if ch != nil {
			defer close(ch)
		}

		// get the cluster info
		clusterInfo, err := client.GetClusterInfo()
		if err != nil {
			log.Fatal(err, "failed to get vineyard cluster info")
		}
		output := util.NewOutput(nil, nil, &clusterInfo)
		// set the output options
		output.WithFilter(false).
			SortedKey(flags.SortedKey).
			SetFormat(flags.Format)

		Output = output
	},
}

func NewGetClusterInfoCmd() *cobra.Command {
	return getClusterInfo
}

func init() {
	flags.ApplyConnectOpts(getClusterInfo)
	flags.ApplyOutputOpts(getClusterInfo)
}
