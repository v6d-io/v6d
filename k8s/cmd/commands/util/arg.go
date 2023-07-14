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
package util

import (
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

func AssertNoArgs(cmd *cobra.Command, args []string) {
	if err := cobra.NoArgs(cmd, args); err != nil {
		log.Fatal(err, "Expects no positional arguments")
	}
}

func AssertNoArgsOrInput(cmd *cobra.Command, args []string) {
	if err := cobra.NoArgs(cmd, args); err != nil {
		if args[0] == "-" {
			return
		}
		log.Fatal(err, "Expects no positional arguments")
	}
}

func CheckClientSocket(cmd *cobra.Command, args []string) {
	ipcSocket := flags.GetIPCSocket()
	rpcSocket := flags.GetRPCSocket()
	vineyardDeployment := flags.DeploymentName
	if ipcSocket == "" && rpcSocket == "" && vineyardDeployment == "" {
		log.Fatal(nil, "Please specify the ipc socket or rpc socket")
	}
}
