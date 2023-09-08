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
	"syscall"

	arrow "github.com/apache/arrow/go/v11/arrow/memory"
	"github.com/pkg/errors"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client/io"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/memory"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type IPCClient struct {
	*ClientBase
	unixConn  *net.UnixConn
	mmapTable map[int]MmapEntry
}

type MmapEntry struct {
	FD   int
	Size int64
	ro   []byte // slice
	rw   []byte // slice
}

func (m *MmapEntry) mmap(readonly bool) error {
	protection := syscall.PROT_READ
	if !readonly {
		protection |= syscall.PROT_WRITE
	}
	data, err := syscall.Mmap(m.FD, 0, int(m.Size), protection, syscall.MAP_SHARED)
	if err == nil {
		if readonly {
			m.ro = data
		} else {
			m.rw = data
		}
	}
	return err
}

// Connect to IPCClient steps as follows
//
// 1. using unix socket connect to vineyard server
// 2. sending register request to server and get response from server
//
// Note: you should send message's length first to server, then send message
func NewIPCClient(socket string) (*IPCClient, error) {
	c := &IPCClient{}
	c.ClientBase = &ClientBase{}
	c.IPCSocket = socket

	c.unixConn = new(net.UnixConn)
	if err := io.ConnectIPCSocketRetry(socket, &c.unixConn); err != nil {
		return nil, err
	}
	c.conn = c.unixConn

	messageOut := common.WriteRegisterRequest(common.VINEYARD_VERSION_STRING)
	if err := c.doWrite(messageOut); err != nil {
		return nil, err
	}
	var reply common.RegisterReply
	if err := c.doReadReply(&reply); err != nil {
		return nil, err
	}

	c.connected = true
	c.RPCEndpoint = reply.RPCEndpoint
	c.InstanceID = reply.InstanceID
	c.serverVersion = reply.Version

	c.mmapTable = make(map[int]MmapEntry)
	return c, nil
}

func (c *IPCClient) CreateBuffer(size uint64) (blob BlobWriter, err error) {
	if size == 0 {
		return EmptyBlobWriter(c), nil
	}

	messageOut := common.WriteCreateBufferRequest(size)
	if err := c.doWrite(messageOut); err != nil {
		return blob, err
	}
	var reply common.CreateBufferReply
	if err := c.doReadReply(&reply); err != nil {
		return blob, err
	}
	payload := reply.Created
	if size != payload.DataSize {
		return blob, errors.New("data size not match")
	}

	pointer, err := c.mmapToClient(payload.StoreFd, int64(payload.MapSize), false, true)
	if err != nil {
		return blob, err
	}
	v := memory.Slice(pointer, payload.DataOffset, payload.DataSize)
	blob.Reset(payload.ID, payload.DataSize, arrow.NewBufferBytes(v), c.InstanceID)
	return blob, nil
}

func (c *IPCClient) BuildBuffer(address []byte, size uint64) (types.ObjectID, error) {
	if size == 0 {
		empty_blob_writer := EmptyBlobWriter(c)
		return empty_blob_writer.Seal(c)
	}
	if address == nil {
		empty_blob_writer := EmptyBlobWriter(c)
		return empty_blob_writer.Seal(c)
	}
	messageOut := common.WriteCreateBufferRequest(size)
	if err := c.doWrite(messageOut); err != nil {
		return uint64(0), err
	}
	var reply common.CreateBufferReply
	if err := c.doReadReply(&reply); err != nil {
		return uint64(0), err
	}
	payload := reply.Created
	if size != payload.DataSize {
		return uint64(0), errors.New("data size not match")
	}
	pointer, err := c.mmapToClient(payload.StoreFd, int64(payload.MapSize), false, true)
	if err != nil {
		return uint64(0), err
	}
	v := memory.Slice(pointer, payload.DataOffset, payload.DataSize)
	copy(v, address)
	buffer := EmptyBlobWriter(c)
	buffer.Reset(payload.ID, payload.DataSize, arrow.NewBufferBytes(v), c.InstanceID)
	//fmt.Println(buffer.Buffer.Buf())
	return buffer.Seal(c)
}

func (c *IPCClient) GetBuffer(id types.ObjectID, unsafe bool) (Blob, error) {
	buffers, err := c.GetBuffers([]types.ObjectID{id}, unsafe)
	if err != nil {
		return Blob{}, err
	}
	return buffers[id], nil
}

func (c *IPCClient) GetBuffers(
	ids []types.ObjectID,
	unsafe_ bool,
) (map[types.ObjectID]Blob, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	messageOut := common.WriteGetBuffersRequest(ids, unsafe_)
	if err := c.doWrite(messageOut); err != nil {
		return nil, err
	}
	var reply common.GetBuffersReply
	if err := c.doReadReply(&reply); err != nil {
		return nil, err
	}
	buffers := make(map[types.ObjectID]Blob)
	for _, payload := range reply.Payloads {
		if payload.DataSize == 0 {
			buffers[payload.ID] = Blob{Object{payload.ID, nil}, 0, nil}
			continue
		}
		pointer, err := c.mmapToClient(payload.StoreFd, int64(payload.MapSize), true, true)
		if err != nil {
			return nil, err
		}
		v := memory.Slice(pointer, payload.DataOffset, payload.DataSize)
		buffers[payload.ID] = Blob{
			Object{payload.ID, nil},
			payload.DataSize,
			arrow.NewBufferBytes(v),
		}
	}
	return buffers, nil
}

func (c *IPCClient) CreateMetaData(metadata *ObjectMeta) (id types.ObjectID, err error) {
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

func (c *IPCClient) GetMetaData(id types.ObjectID, syncRemote bool) (meta *ObjectMeta, err error) {
	if !c.connected {
		return nil, NOT_CONNECTED_ERR
	}
	metadatas, err := c.GetData([]types.ObjectID{id}, syncRemote, false)
	if err != nil {
		return nil, err
	}
	meta = NewObjectMeta()
	meta.Reset()
	meta.SetMetaData(c.ClientBase, metadatas[0])

	bufferset := meta.GetBuffers()
	buffers, err := c.GetBuffers(bufferset.GetBufferIds(), false)
	if err != nil {
		return nil, err
	}
	for id, buffer := range buffers {
		_ = bufferset.EmplaceBuffer(id, buffer.Buffer)
	}
	return meta, nil
}

func (c *IPCClient) GetObject(id types.ObjectID, object IObject) error {
	if !c.connected {
		return NOT_CONNECTED_ERR
	}
	meta, err := c.GetMetaData(id, false)
	if err != nil {
		return err
	}
	return object.Construct(c, meta)
}

func (c *IPCClient) GetClusterInfo() (map[string]any, error) {
	if !c.connected {
		return nil, NOT_CONNECTED_ERR
	}
	return c.GetClusterMeta()
}

func (c *IPCClient) ListMetadata(pattern string, regex bool, limit int) (map[string]map[string]any, error) {
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

func (c *IPCClient) mmapToClient(
	fd int,
	mapSize int64,
	readOnly bool,
	realign bool,
) ([]byte, error) {
	if _, ok := c.mmapTable[fd]; !ok {
		file, err := c.unixConn.File()
		if err != nil {
			return nil, errors.Wrap(err, "failed to obtain native socket")
		}
		clientFd, err := memory.RecvFileDescriptor(int(file.Fd()))
		if err != nil {
			return nil, err
		}
		if clientFd <= 0 {
			return nil, errors.New("receive client fd error")
		}
		entry := MmapEntry{FD: clientFd}
		if realign {
			entry.Size = mapSize + 8 /* sizeof(size_t) */
		} else {
			entry.Size = mapSize
		}
		c.mmapTable[fd] = entry
	}
	entry := c.mmapTable[fd]
	if readOnly && entry.ro == nil || !readOnly && entry.rw == nil {
		if err := entry.mmap(readOnly); err != nil {
			return nil, err
		}
	}
	if readOnly {
		return entry.ro, nil
	} else {
		return entry.rw, nil
	}
}
