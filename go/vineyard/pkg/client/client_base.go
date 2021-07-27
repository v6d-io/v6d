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
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"net"
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
