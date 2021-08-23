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
	PUT_NAME_REQUEST       = "put_name_request"
	PUT_NAME_REPLY         = "put_name_reply"
	GET_NAME_REQUEST       = "get_name_request"
	GET_NAME_REPLY         = "get_name_reply"
	DROP_NAME_REQUEST      = "drop_name_request"
	DROP_NAME_REPLY        = "drop_name_reply"
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
	Type string   `json:"type"`
	ID   ObjectID `json:"id"`
}

type PersisReply struct {
	Type string `json:"type"`
	Code int    `json:"code"`
}

type PutNameRequest struct {
	Type        string   `json:"type"`
	ReqObjectID ObjectID `json:"object_id"`
	Name        string   `json:"name"`
}

type GetNameRequest struct {
	Type string `json:"type"`
	Name string `json:"name"`
	Wait bool   `json:"wait"`
}

type PutNameReply struct {
	Type string `json:"type"`
	Code int    `json:"code"`
}

type GetNameReply struct {
	Type        string   `json:"type"`
	Code        int      `json:"code"`
	RepObjectID ObjectID `json:"object_id"`
}

type DropNameRequest struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type DropNameReply struct {
	Type string `json:"type"`
	Code int    `json:"code"`
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

func WritePersistRequest(id ObjectID, msg *string) {
	var persist PersistRequest
	persist.Type = PERSIST_REQUEST
	persist.ID = id

	encodeMsg(persist, msg)
}

func WritePutNameRequest(id ObjectID, name string, msg *string) {
	var putNameReq PutNameRequest
	putNameReq.Type = PUT_NAME_REQUEST
	putNameReq.ReqObjectID = id
	putNameReq.Name = name

	encodeMsg(putNameReq, msg)
}

func WriteGetNameRequest(name string, wait bool, msg *string) {
	var getNameReq GetNameRequest
	getNameReq.Type = GET_NAME_REQUEST
	getNameReq.Name = name
	getNameReq.Wait = wait

	encodeMsg(getNameReq, msg)
}

func WriteDropNameRequest(name string, msg *string) {
	var dropNameReq DropNameRequest
	dropNameReq.Type = DROP_NAME_REQUEST
	dropNameReq.Name = name

	encodeMsg(dropNameReq, msg)
}
