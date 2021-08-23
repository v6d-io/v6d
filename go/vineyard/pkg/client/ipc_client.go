/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

package vineyard

import (
	"encoding/json"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"net"
)

type IPCClient struct {
	ClientBase
	connected     bool
	ipcSocket     string
	conn          *net.UnixConn
	instanceID    int
	serverVersion string
	rpcEndpoint   string
}

// Connect to IPCClient steps as follows
// 1. using unix socket connecct to vineyead server
// 2. sending register request to server and get response from server
// Note: you should send message's length first to server, then send message
func (i *IPCClient) Connect(ipcSocket string) error {
	if i.connected || i.ipcSocket == ipcSocket {
		return nil
	}
	i.ipcSocket = ipcSocket
	i.conn = new(net.UnixConn)
	if err := ConnectIPCSocketRetry(i.ipcSocket, &i.conn); err != nil {
		return err
	}
	i.ClientBase.conn = i.conn
	var messageOut string
	common.WriteRegisterRequest(&messageOut)
	if err := i.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	err := i.DoRead(&messageIn)
	if err != nil {
		return err
	}
	var registerReply common.RegisterReply
	err = json.Unmarshal([]byte(messageIn), &registerReply)
	if err != nil {
		return err
	}
	i.instanceID = registerReply.InstanceID
	if registerReply.Version == "" {
		i.serverVersion = common.DEFAULT_SERVER_VERSION
	} else {
		i.serverVersion = registerReply.Version
	}
	i.connected = true
	i.rpcEndpoint = registerReply.RPCEndpoint
	// TODO: compatible server check
	return nil
}

func (i *IPCClient) CreateBlob(size int, ) {
	if i.connected == false {
		return
	}
	//var id common.ObjectID = common.InvalidObjectID()

}
