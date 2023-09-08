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
package flags

import (
	"os"

	"github.com/spf13/cobra"
)

// connect options
var (
	// IPCSocket represents the vineyard IPC socket path
	IPCSocket string

	// RPCSocket represents the vineyard RPC socket path
	RPCSocket string

	// DeploymentName is the name of vineyard deployment
	DeploymentName string

	// Port represents the port of vineyard deployment
	Port int

	// ForwardPort is the forward port of vineyard deployment
	ForwardPort int
)

// get options
var (
	// The object id to get
	Object_id string

	// If the target object is a remote object, code_remote=True will force a meta synchronization on the vineyard server.
	SyncRemote bool

	// unsafe means getting the blob even the blob is not sealed yet. Default is False.
	Unsafe bool
)

// put options
var (
	// The value to put
	Value string
)

// ls options
var (
	// Pattern string that will be matched against the object’s typenames
	Pattern string

	// Regex represents whether the pattern string will be considered as a regex expression
	Regex bool

	// Limit represents the maximum number of objects to return
	Limit int

	// SortedKey represents the key to sort the objects
	SortedKey string
)

// output options
var (
	// Format represents the output format, support table or json, default is table
	Format string
)

// GetIPCSocket returns the vineyard IPC socket path
func GetIPCSocket() string {
	if IPCSocket == "" {
		return os.Getenv("VINEYARD_IPC_SOCKET")
	}
	return IPCSocket
}

// GetRPCSocket returns the vineyard RPC socket path
func GetRPCSocket() string {
	if RPCSocket == "" {
		return os.Getenv("VINEYARD_RPC_SOCKET")
	}
	return RPCSocket
}

func ApplyLimitOpt(cmd *cobra.Command) {
	cmd.Flags().IntVarP(&Limit, "limit", "l", 5, "maximum number of objects to return")
}

func ApplyGetOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&Object_id, "object_id", "", "", "The object id to get")
	cmd.Flags().BoolVarP(&SyncRemote, "syncRemote", "", false, "If the target object is a remote object,"+
		"code_remote=True will force a meta synchronization on the vineyard server.")
	cmd.Flags().BoolVarP(&Unsafe, "unsafe", "", false, "unsafe means getting the blob even the blob is not sealed yet."+
		"Default is False.")
}

func ApplyPutOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&Value, "value", "", "", "vineyard blob value")
}

func ApplyLsOpts(cmd *cobra.Command) {
	ApplyLimitOpt(cmd)
	cmd.Flags().StringVarP(&Pattern, "pattern", "p", "*", "string that will be matched against the object’s typenames")
	cmd.Flags().BoolVarP(&Regex, "regex", "r", false, "regex pattern to match the object’s typenames")
	cmd.Flags().StringVarP(&SortedKey, "sorted-key", "k", "id",
		"key to sort the objects, support:"+"\n"+
			"- id: object id, the default value."+"\n"+
			"- typename: object typename, e.g. tensor, dataframe, etc."+"\n"+
			"- type: object type, e.g. global, local, etc."+"\n"+
			"- instance_id: object instance id.")
}

// ApplyOutputOpts applies the output options
func ApplyOutputOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&Format, "format", "o", "table",
		"the output format, support table or json, default is table")
}

// ApplyConnectOpts applies the connect options
func ApplyConnectOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&IPCSocket, "ipc-socket", "", "", "vineyard IPC socket path")
	cmd.Flags().StringVarP(&RPCSocket, "rpc-socket", "", "", "vineyard RPC socket path")
	cmd.Flags().StringVarP(&DeploymentName, "deployment-name", "", "", "the name of vineyard deployment")
	cmd.Flags().IntVarP(&Port, "port", "", 9600, "the port of vineyard deployment")
	cmd.Flags().IntVarP(&ForwardPort, "forward-port", "", 9600, "the forward port of vineyard deployment")
}
