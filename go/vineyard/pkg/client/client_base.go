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
	"errors"
	"net"

	vineyard "github.com/v6d-io/v6d/go/vineyard/pkg/client/ds"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
)

type Signature = uint64

type ClientBase struct {
	// TODO: fix unify connection conn
	conn       net.Conn
	connected  bool
	instanceID common.InstanceID
}

func (c *ClientBase) DoWrite(msgOut string) error {
	err := SendMessage(c.conn, msgOut)
	if err != nil {
		c.connected = false
	}
	return nil
}

func (c *ClientBase) DoRead(msg *string) error {
	return RecvMessage(c.conn, msg)
}

func (c *ClientBase) Disconnect() error {
	if !c.connected {
		return nil
	}
	var messageOut string
	common.WriteExitRequest(&messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	c.conn.Close()
	c.connected = false
	return nil
}

func (c *ClientBase) Persist(id common.ObjectID) error {
	var messageOut string
	common.WritePersistRequest(id, &messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	if err := c.DoRead(&messageIn); err != nil {
		return err
	}

	var persistReply common.PersisReply
	err := json.Unmarshal([]byte(messageIn), &persistReply)
	if err != nil {
		return err
	}

	if persistReply.Code != 0 || persistReply.Type != common.PERSIST_REQUEST {
		return errors.New("get persist response from vineyard failed")
	}
	return nil
}

func (c *ClientBase) PutName(id common.ObjectID, name string) error {
	var messageOut string
	common.WritePutNameRequest(id, name, &messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	if err := c.DoRead(&messageIn); err != nil {
		return err
	}

	var putNameReply common.PutNameReply
	err := json.Unmarshal([]byte(messageIn), &putNameReply)
	if err != nil {
		return err
	}

	if putNameReply.Code != 0 || putNameReply.Type != common.PUT_NAME_REPLY {
		return &common.ReplyError{Code: putNameReply.Code, Type: putNameReply.Type, Err: err}
	}
	return nil
}

func (c *ClientBase) GetName(name string, wait bool, id *common.ObjectID) error {
	var messageOut string
	common.WriteGetNameRequest(name, wait, &messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	if err := c.DoRead(&messageIn); err != nil {
		return err
	}

	var getNameReply common.GetNameReply
	err := json.Unmarshal([]byte(messageIn), &getNameReply)
	if err != nil {
		return err
	}

	if getNameReply.Code != 0 || getNameReply.Type != common.GET_NAME_REPLY {
		return &common.ReplyError{Code: getNameReply.Code, Type: getNameReply.Type, Err: err}
	}
	*id = getNameReply.RepObjectID
	return nil
}

func (c *ClientBase) DropName(name string) error {
	var messageOut string
	common.WriteDropNameRequest(name, &messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	if err := c.DoRead(&messageIn); err != nil {
		return err
	}

	var dropNameReply common.DropNameReply
	err := json.Unmarshal([]byte(messageIn), &dropNameReply)
	if err != nil {
		return err
	}

	if dropNameReply.Code != 0 || dropNameReply.Type != common.DROP_NAME_REPLY {
		return &common.ReplyError{Code: dropNameReply.Code, Type: dropNameReply.Type, Err: err}
	}
	return nil
}

func (c *ClientBase) GetData(id common.ObjectID, getDataReply *common.GetDataReply, syncRemote, wait bool) error {
	if !c.connected {
		return nil
	}
	var messageOut string
	common.WriteGetDataRequest(id, syncRemote, wait, &messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	if err := c.DoRead(&messageIn); err != nil {
		return err
	}
	err := json.Unmarshal([]byte(messageIn), getDataReply)
	if err != nil {
		return err
	}
	return nil
}

func (c *ClientBase) SyncMetaData() error {
	var getDataReply common.GetDataReply
	return c.GetData(common.InvalidObjectID(), &getDataReply, true, false)
}

func (c *ClientBase) CreateData(tree interface{}, id *common.ObjectID, signature *Signature, instanceID *common.InstanceID) error {
	if !c.connected {
		return nil
	}
	var messageOut string
	common.WriteCreateDataRequest(id, &messageOut)
	if err := c.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	if err := c.DoRead(&messageIn); err != nil {
		return err
	}
	var createDataReply common.CreateDataReply
	err := json.Unmarshal([]byte(messageIn), &createDataReply)
	if err != nil {
		return err
	}
	id = &createDataReply.ID
	signature = &createDataReply.Signature
	instanceID = &createDataReply.InstanceID
	return nil
}

func (c *ClientBase) GetMetaData(id common.ObjectID, meta *vineyard.ObjectMeta, syncRemote bool) {
	if !c.connected {
		return
	}
	var getDataReply common.GetDataReply
	err := c.GetData(id, &getDataReply, syncRemote, false)
	if err != nil {
		return
	}
	meta.Reset()
	//meta.SetMetaData(*i, getDataReply) // TODO: how to set
	// TODO
	return
}

func (c *ClientBase) CreateMetaData(metaData *vineyard.ObjectMeta, id common.ObjectID) {
	var instanceID common.InstanceID = c.instanceID
	metaData.Init()
	metaData.SetInstanceId(instanceID)
	if !metaData.HasKey("nbytes") {
		metaData.SetNBytes(0)
	}
	if metaData.InComplete() {
		c.SyncMetaData() // TODO: check correctness
	}
	var signature Signature
	err := c.CreateData(metaData.MetaData(), &id, &signature, &instanceID)
	if err != nil {
		metaData.SetId(id)
		metaData.SetSignature(signature)
		// metaData.SetClient(c) // TODO: fix type not match
		metaData.SetInstanceId(instanceID)
		if metaData.InComplete() {
			var resultMeta vineyard.ObjectMeta
			c.GetMetaData(id, &resultMeta, false)
			metaData = &resultMeta
		}
	}
	return
}
