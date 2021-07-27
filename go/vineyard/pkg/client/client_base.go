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
	"errors"
	"net"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
)

type ClientBase struct {
	// TODO: fix unify connection conn
	conn      net.Conn
	connected bool
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
	err := json.Unmarshal([]byte(messageIn), persistReply)
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
