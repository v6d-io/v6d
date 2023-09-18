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

// Package csi contains the start command of vineyard csi driver
package csi

import (
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/csidriver"
)

var csiExample = util.Examples(`
	# start the csi with the specific endpoint and node id
	vineyardctl csi --endpoint=unix:///csi/csi.sock --nodeid=csinode1`)

// csiCmd starts the vineyard csi driver
var csiCmd = &cobra.Command{
	Use:     "csi",
	Short:   "Start the vineyard csi driver",
	Example: csiExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		d := csidriver.NewDriver(flags.NodeID, flags.Endpoint)
		d.Run()
	},
}

func NewCsiCmd() *cobra.Command {
	return csiCmd
}

func init() {
	flags.ApplyCsiOpts(csiCmd)
}
