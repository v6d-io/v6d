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
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client/io"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type RPCClient struct {
	*ClientBase
	remoteInstanceID types.InstanceID
}

func NewRPCClient(rpcEndpoint string) (*RPCClient, error) {
	c := &RPCClient{}
	c.ClientBase = &ClientBase{}
	c.RPCEndpoint = rpcEndpoint

	addresses := strings.Split(rpcEndpoint, ":")
	var port uint16
	if addresses[1] == "" {
		_, port = GetDefaultRPCHostAndPort()
	} else {
		if p, err := strconv.Atoi(addresses[1]); err != nil {
			_, port = GetDefaultRPCHostAndPort()
		} else {
			port = uint16(p)
		}
	}

	var conn net.Conn
	if err := io.ConnectRPCSocketRetry(addresses[0], port, &conn); err != nil {
		return nil, err
	}
	c.conn = conn

	messageOut := common.WriteRegisterRequest(common.VINEYARD_VERSION_STRING)
	if err := c.doWrite(messageOut); err != nil {
		return nil, err
	}
	var reply common.RegisterReply
	if err := c.doReadReply(&reply); err != nil {
		return nil, err
	}

	c.connected = true
	c.IPCSocket = reply.IPCSocket
	c.InstanceID = types.UnspecifiedInstanceID()
	c.serverVersion = reply.Version

	c.remoteInstanceID = reply.InstanceID
	return c, nil
}

func (c *RPCClient) CreateMetaData(metadata *ObjectMeta) (id types.ObjectID, err error) {
	if !c.connected {
		return types.InvalidObjectID(), NOT_CONNECTED_ERR
	}

	metadata.SetInstanceId(c.InstanceID)
	metadata.SetTransient(true)
	if !metadata.HasKey("nbytes") {
		metadata.SetNBytes(0)
	}
	if metadata.InComplete() {
		_ = c.SyncMetaData()
	}

	id, signature, instanceId, err := c.CreateData(metadata.MetaData())
	if err != nil {
		return id, err
	}

	metadata.SetId(id)
	metadata.SetSignature(signature)
	metadata.SetInstanceId(instanceId)
	if metadata.InComplete() {
		if meta, err := c.GetMetaData(id, false); err != nil {
			return types.InvalidObjectID(), err
		} else {
			*metadata = *meta
		}
	}
	return id, nil
}

func (c *RPCClient) GetMetaData(id types.ObjectID, syncRemote bool) (meta *ObjectMeta, err error) {
	if !c.connected {
		return nil, NOT_CONNECTED_ERR
	}
	metadatas, err := c.GetData([]types.ObjectID{id}, syncRemote, false)
	if err != nil {
		return nil, err
	}
	meta = NewObjectMeta()
	meta.Reset()
	meta.SetMetaData(nil, metadatas[0])
	return meta, nil
}

func (c *RPCClient) GetClusterInfo() (map[string]any, error) {
	if !c.connected {
		return nil, NOT_CONNECTED_ERR
	}
	return c.GetClusterMeta()
}

func (c *RPCClient) ListMetaData(pattern string, regex bool, limit int) (map[string]map[string]any, error) {
	if !c.connected {
		return nil, NOT_CONNECTED_ERR
	}
	messageOut := common.WriteListDataRequest(pattern, regex, limit)
	if err := c.doWrite(messageOut); err != nil {
		return nil, err
	}
	var reply common.ListDataReply
	if err := c.doReadReply(&reply); err != nil {
		return nil, err
	}

	return reply.Content, nil
}
