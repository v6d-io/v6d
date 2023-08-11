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

package client

import (
	"net"
	"os"
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client/io"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/log"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

const (
	VINEYARD_IPC_SOCKET_KEY   = "VINEYARD_IPC_SOCKET"
	VINEYARD_RPC_ENDPOINT_KEY = "VINEYARD_RPC_ENDPOINT"
	VINEYARD_DEFAULT_RPC_PORT = 9600
)

var logger = log.Log.WithName("client")

var NOT_CONNECTED_ERR = common.NotConnected()

func GetDefaultIPCSocket() string {
	return os.Getenv(VINEYARD_IPC_SOCKET_KEY)
}

func GetDefaultRPCEndpoint() string {
	return os.Getenv(VINEYARD_RPC_ENDPOINT_KEY)
}

func GetDefaultRPCHostAndPort() (string, uint16) {
	rpcEndpoint := GetDefaultRPCEndpoint()
	parts := strings.Split(rpcEndpoint, ":")
	if len(parts) == 1 {
		return rpcEndpoint, VINEYARD_DEFAULT_RPC_PORT
	}
	port, err := strconv.Atoi(parts[1])
	if err != nil {
		return parts[0], VINEYARD_DEFAULT_RPC_PORT
	}
	return parts[0], uint16(port)
}

type ClientBase struct {
	conn          net.Conn
	connected     bool
	IPCSocket     string
	RPCEndpoint   string
	serverVersion string
	InstanceID    types.InstanceID
}

func (c *ClientBase) doWrite(message []byte) error {
	err := io.SendMessageBytes(c.conn, message)
	if err != nil {
		c.connected = false
	}
	return err
}

func (c *ClientBase) doRead() ([]byte, error) {
	return io.RecvMessageBytes(c.conn)
}

func (c *ClientBase) doReadReply(v common.Reply) error {
	messageIn, err := c.doRead()
	if err != nil {
		return err
	}
	if err := common.ParseJson(messageIn, v); err != nil {
		return err
	}
	return v.Check()
}

func (c *ClientBase) Disconnect() {
	if !c.connected {
		return
	}
	messageOut := common.WriteExitRequest()
	if err := c.doWrite(messageOut); err != nil {
		log.Error(err, "failed to disconnect the client")
	}
	c.connected = false
	_ = c.conn.Close() // ignore the error in `close`
}

func (c *ClientBase) Seal(id types.ObjectID) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WriteSealRequest(id)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.SealReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) DropBuffer(id types.ObjectID) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WriteDropBufferRequest(id)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.DropBufferReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) IncreaseRefCount(ids []types.ObjectID) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WriteIncreaseRefCountRequest(ids)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.IncreaseRefCountReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) Release(id types.ObjectID) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WriteReleaseRequest(id)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.ReleaseReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) CreateData(
	tree map[string]any,
) (id types.ObjectID, signature types.Signature, instanceID types.InstanceID, err error) {
	if !c.connected {
		return id, signature, instanceID, NOT_CONNECTED_ERR
	}
	messageOut := common.WriteCreateDataRequest(tree)
	if err := c.doWrite(messageOut); err != nil {
		return id, signature, instanceID, err
	}
	var reply common.CreateDataReply
	if err := c.doReadReply(&reply); err != nil {
		return id, signature, instanceID, err
	}
	id = reply.ID
	signature = reply.Signature
	instanceID = reply.InstanceID
	return id, signature, instanceID, nil
}

func (c *ClientBase) GetData(
	ids []types.ObjectID,
	syncRemote bool,
	wait bool,
) (objects []map[string]any, err error) {
	messageOut := common.WriteGetDataRequest(ids, syncRemote, wait)
	if err := c.doWrite(messageOut); err != nil {
		return nil, err
	}
	var reply common.GetDataReply
	if err := c.doReadReply(&reply); err != nil {
		return nil, err
	}
	objects = make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		if item, ok := reply.Content[types.ObjectIDToString(id)]; ok {
			objects = append(objects, item)
		} else {
			objects = append(objects, nil)
		}
	}
	return objects, nil
}

func (c *ClientBase) GetClusterMeta() (map[string]any, error) {
	if !c.connected {
		return nil, NOT_CONNECTED_ERR
	}
	messageOut := common.WriteClusterMetaRequest()
	if err := c.doWrite(messageOut); err != nil {
		return nil, err
	}
	var reply common.ClusterMetaReply
	if err := c.doReadReply(&reply); err != nil {
		return nil, err
	}
	return reply.Meta.(map[string]any), nil
}

func (c *ClientBase) Delete(ids []types.ObjectID) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WriteDeleteDataRequest(ids, false)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.DeleteDataReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) SyncMetaData() error {
	_, err := c.GetData([]types.ObjectID{types.InvalidObjectID()}, true, false)
	return err
}

func (c *ClientBase) Persist(id types.ObjectID) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WritePersistRequest(id)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.PersistReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) PutName(id types.ObjectID, name string) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WritePutNameRequest(id, name)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.PutNameReply
	return c.doReadReply(&reply)
}

func (c *ClientBase) GetName(name string, wait bool) (id types.ObjectID, err error) {
	if !c.connected {
		return id, NOT_CONNECTED_ERR
	}
	messageOut := common.WriteGetNameRequest(name, wait)
	if err = c.doWrite(messageOut); err != nil {
		return id, err
	}
	var reply common.GetNameReply
	if err = c.doReadReply(&reply); err != nil {
		return id, err
	}
	id = reply.ID
	return id, err
}

func (c *ClientBase) DropName(name string) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	messageOut := common.WriteDropNameRequest(name)
	if err := c.doWrite(messageOut); err != nil {
		return err
	}
	var reply common.DropNameReply
	return c.doReadReply(&reply)
}
