/** Copyright 2020-2023 Alibaba Group Holding Limited.

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
	"net"
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
)

type RPCClient struct {
	ClientBase
	connected        bool
	ipcSocket        string
	conn             *net.UnixConn
	instanceID       int
	remoteInstanceID int
	rpcEndpoint      string
}

func (r *RPCClient) Connect(rpcEndpoint string) error {
	if r.connected || r.rpcEndpoint == rpcEndpoint {
		return nil
	}
	r.rpcEndpoint = rpcEndpoint

	str := strings.Split(rpcEndpoint, ":")
	host := str[0]
	port := str[1]
	if port == "" {
		port = "9600"
	}
	var conn net.Conn
	portNum, err := strconv.Atoi(port)
	if err != nil {
		return err
	}
	err = ConnectRPCSocketRetry(host, uint16(portNum), &conn)
	if err != nil {
		return err
	}

	r.ClientBase.conn = conn
	var messageOut string
	common.WriteRegisterRequest(&messageOut)
	if err := r.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	err = r.DoRead(&messageIn)
	if err != nil {
		return err
	}
	var registerReply common.RegisterReply
	err = json.Unmarshal([]byte(messageIn), &registerReply)
	if err != nil {
		return err
	}

	r.connected = true
	r.ipcSocket = registerReply.IPCSocket
	r.remoteInstanceID = registerReply.InstanceID
	// TODO: compatible server check

	// r.instanceID = registerReply.InstanceID
	// fmt.Println(messageIn)
	return nil
}
