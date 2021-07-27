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

package common

import (
	"encoding/json"
)

const (
	REGISTER_REQUEST       = "register_request"
	REGISTER_REPLY         = "register_reply"
	EXIT_REQUEST           = "exit_request"
	PERSIST_REQUEST        = "persist_request"
	DEFAULT_SERVER_VERSION = "0.0.0"
)

type RegisterRequest struct {
	Type    string `json:"type"`
	Version string `json:"version"`
}

type RegisterReply struct {
	InstanceID  int    `json:"instance_id"`
	IPCSocket   string `json:"ipc_socket"`
	RPCEndpoint string `json:"rpc_endpoint"`
	Type        string `json:"type"`
	Version     string `json:"version,omitempty"`
}

type ExitRequest struct {
	Type string `json:"type"`
}

type PersistRequest struct {
	Type string `json:"type"`
	ID   int    `json:"id"`
}

func encodeMsg(data interface{}, msg *string) error {
	msgBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}
	*msg = string(msgBytes)
	return nil
}

func WriteRegisterRequest(msg *string) {
	var register RegisterRequest
	register.Type = REGISTER_REQUEST
	register.Version = "0.2.4"

	encodeMsg(register, msg)
}

func WriteExitRequest(msg *string) {
	var exit ExitRequest
	exit.Type = EXIT_REQUEST

	encodeMsg(exit, msg)
}

func WritePersistRequest(id int, msg *string) {
	var persist PersistRequest
	persist.Type = PERSIST_REQUEST
	persist.ID = id

	encodeMsg(persist, msg)
}
