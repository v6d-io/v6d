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

/*
#cgo CFLAGS: -I ../common/memory
#cgo LDFLAGS: -L ../common/memory -lfling

#include <sys/mman.h>

#include "fling.h"
*/
// nolint: typecheck
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"unsafe"

	"github.com/apache/arrow/go/arrow/memory"
	"github.com/v6d-io/v6d/go/vineyard/pkg/client/ds"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
)

type IPCClient struct {
	ClientBase
	connected     bool
	ipcSocket     string
	conn          *net.UnixConn
	instanceID    int
	serverVersion string
	rpcEndpoint   string
	mmapTable     map[int]MmapEntry
}

type MmapEntry struct {
	clientFd  int
	mapSize   int64
	readOnly  bool
	realign   bool
	roPointer unsafe.Pointer
	rwPointer unsafe.Pointer
}

func (m *MmapEntry) MapReadOnly() {
	m.roPointer = C.mmap(nil, C.ulong(m.mapSize), C.PROT_READ, C.MAP_SHARED, C.int(m.clientFd), 0)
	// TODO: error fix
}

func (m *MmapEntry) MapReadWrite() {
	m.rwPointer = C.mmap(nil, C.ulong(m.mapSize), C.PROT_READ|C.PROT_WRITE, C.MAP_SHARED, C.int(m.clientFd), 0)
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
	i.mmapTable = make(map[int]MmapEntry)
	// TODO: compatible server check
	return nil
}

func (i *IPCClient) CreateBlob(size int, blob *ds.BlobWriter) {
	if i.connected == false {
		return
	}
	var buffer memory.Buffer
	var id common.ObjectID = common.InvalidObjectID()
	var payload ds.Payload
	i.CreateBuffer(size, &id, &payload, &buffer)
	blob.Reset(id, payload, buffer)
}

func (i *IPCClient) CreateBuffer(size int, id *common.ObjectID, payload *ds.Payload, buffer *memory.Buffer) error {
	if i.connected == false {
		return errors.New("ipc client is not connected")
	}
	var messageOut string
	common.WriteCreateBufferRequest(size, &messageOut)

	if err := i.DoWrite(messageOut); err != nil {
		return err
	}
	var messageIn string
	err := i.DoRead(&messageIn)
	if err != nil {
		return err
	}
	fmt.Println("receive from vineyard create buffer is :", messageIn)
	var createBufferReply common.CreateBufferReply
	err = json.Unmarshal([]byte(messageIn), &createBufferReply)
	if err != nil {
		fmt.Println("create buffer reply json failed")
		return err
	}
	*id = createBufferReply.ID // TODO: check whether two id is same
	payload.ID = createBufferReply.ID
	payload.StoreFd = createBufferReply.Created.StoreFd
	payload.DataOffset = createBufferReply.Created.DataOffset
	payload.DataSize = createBufferReply.Created.DataSize
	payload.MapSize = createBufferReply.Created.MapSize

	if size != payload.DataSize {
		return errors.New("data size not match")
	}

	var shared *uint8
	if payload.DataSize > 0 {
		i.MmapToClient(payload.StoreFd, int64(payload.MapSize), false, true, &shared)
		//i.MmapToClient()
	}
	//fmt.Println(shared[0:1])
	//buffer := memory.NewBufferBytes(shared[])
	return nil
}

func (i *IPCClient) MmapToClient(fd int, mapSize int64, readOnly bool, realign bool, ptr **uint8) error {
	_, ok := i.mmapTable[fd]
	if !ok {
		file, err := i.conn.File()
		if err != nil {
			fmt.Println("Get connection file")
			return err
		}
		clientFd := C.recv_fd(C.int(file.Fd()))
		if clientFd <= 0 {
			return errors.New("receive client fd error")
		}
		newEntry := MmapEntry{int(clientFd), mapSize, readOnly, realign, nil, nil}

		if readOnly {
			newEntry.MapReadOnly()
		} else {
			newEntry.MapReadWrite()
		}
		i.mmapTable[fd] = newEntry
	}

	// TODO: set entry to read only
	//if readOnly {
	//} else {
	//}
	return nil
}
