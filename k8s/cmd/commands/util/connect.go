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
	"strconv"

	"github.com/pkg/errors"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

type Client struct {
	ipcClient *client.IPCClient
	rpcClient *client.RPCClient
}

// NewClient returns a new client connected to the vineyardd server
func NewClient() (Client, chan struct{}) {
	client := Client{}
	var stopChannel chan struct{}
	if ipcSocket := flags.GetIPCSocket(); ipcSocket != "" {
		ipcClient, err := ConnectViaIPC(ipcSocket)
		if err != nil {
			log.Fatal(err, "failed to connect via ipc")
		}
		client.ipcClient = ipcClient
	}
	if rpcSocket := flags.GetRPCSocket(); rpcSocket != "" {
		rpcClient, err := ConnectViaRPC(rpcSocket)
		if err != nil {
			log.Fatal(err, "failed to connect via rpc")
		}
		client.rpcClient = rpcClient
	} else if flags.DeploymentName != "" && flags.Namespace != "" {
		log.Info("Connecting to vineyardd deployment via rpc...")
		stopChannel = make(chan struct{})
		readyChannel := make(chan struct{}, 1)
		rpcClient, err := ConnectDeployment(flags.DeploymentName, flags.Namespace, readyChannel, stopChannel)
		if err != nil {
			log.Fatal(err, "failed to connect vineyard deployment via rpc")
		}
		<-readyChannel
		log.Info("Connected to vineyardd deployment via rpc")
		client.rpcClient = rpcClient
	}
	if client.ipcClient == nil && client.rpcClient == nil {
		log.Fatal(nil, "Unable to connect to vineyardd server")
	}
	return client, stopChannel
}

// ConnectViaIPC connects to the ipc server
func ConnectViaIPC(ipcSocket string) (*client.IPCClient, error) {
	client, err := client.NewIPCClient(ipcSocket)
	if err != nil {
		return nil, errors.Wrap(err, "failed to connect the ipc server")
	}
	return client, nil
}

// ConnectViaRPC connects to the rpc server
func ConnectViaRPC(rpcSocket string) (*client.RPCClient, error) {
	client, err := client.NewRPCClient(rpcSocket)
	if err != nil {
		return nil, errors.Wrap(err, "failed to connect the rpc server")
	}
	return client, nil
}

// ConnectDeployment connects to the vineyardd deployment via rpc
func ConnectDeployment(deployment, namespace string, readyChannel, stopChannel chan struct{}) (*client.RPCClient, error) {
	if deployment != "" && namespace != "" {
		// check if the port is available, if not, generate a new one
		port, err := GetValidForwardPort(flags.ForwardPort)
		if err != nil {
			return nil, errors.Wrap(err, "failed to get a valid forward port")
		}
		flags.ForwardPort = port
		go func() {
			PortForwardDeployment(deployment, namespace, flags.ForwardPort, flags.Port, readyChannel, stopChannel)
		}()
	}
	return ConnectViaRPC("localhost" + ":" + strconv.Itoa(flags.ForwardPort))
}
