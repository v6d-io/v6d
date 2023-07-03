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
	"fmt"
	"unsafe"

	arrow "github.com/apache/arrow/go/v11/arrow/memory"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type Blob struct {
	Object

	Size uint64
	*arrow.Buffer
}

func (b *Blob) Data() ([]byte, error) {
	if b.Size > 0 && b.Buffer.Len() == 0 {
		return nil, fmt.Errorf("The object might be a (partially) remote object "+
			"and the payload data is not locally available: %d", b.Id)
	}
	return b.Buffer.Bytes(), nil
}

func (b *Blob) Pointer() (unsafe.Pointer, error) {
	data, err := b.Data()
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(&data[0]), nil
}

func (b *Blob) Construct(client *IPCClient, meta *ObjectMeta) (err error) {
	if b.Id, err = meta.GetId(); err != nil {
		return err
	}
	b.Meta = meta
	if b.Size, err = meta.GetKeyValueUInt64("length"); err != nil {
		return err
	}
	if b.Buffer, err = meta.GetBuffer(b.Id); err != nil {
		return err
	}
	return nil
}

func EmptyBlob(client *IPCClient) Blob {
	meta := NewObjectMeta()
	meta.SetId(types.EmptyBlobID())
	meta.SetTypeName("vineyard::Blob")
	meta.AddKeyValue("length", 0)
	meta.AddKeyValue("nbytes", 0)
	meta.AddKeyValue("instance_id", client.InstanceID)
	meta.SetTransient(true)
	return Blob{
		Object: Object{
			Id:   types.EmptyBlobID(),
			Meta: meta,
		},
		Size:   0,
		Buffer: nil,
	}
}

type BlobWriter struct {
	Id   types.ObjectID
	Size uint64
	Meta *ObjectMeta

	*arrow.Buffer
}

func (b *BlobWriter) Reset(
	id types.ObjectID,
	size uint64,
	buffer *arrow.Buffer,
	instanceId types.InstanceID,
) {
	b.Id = id
	b.Size = size
	b.Buffer = buffer
	b.Meta = NewObjectMeta()
	b.Meta.SetId(id)
	b.Meta.SetTypeName("vineyard::Blob")
	b.Meta.AddKeyValue("length", size)
	b.Meta.AddKeyValue("nbytes", size)
	b.Meta.AddKeyValue("instance_id", instanceId)
	b.Meta.SetTransient(true)
}

func (b *BlobWriter) Data() ([]byte, error) {
	if b.Size > 0 && b.Buffer.Len() == 0 {
		return nil, fmt.Errorf("The object might be a (partially) remote object "+
			"and the payload data is not locally available: %d", b.Id)
	}
	return b.Buffer.Bytes(), nil
}

func (b *BlobWriter) Pointer() (unsafe.Pointer, error) {
	data, err := b.Data()
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(&data[0]), nil
}

func (b *BlobWriter) Build(client *IPCClient) error {
	return nil
}

func (b *BlobWriter) Seal(client *IPCClient) (types.ObjectID, error) {
	if err := b.Build(client); err != nil {
		return types.InvalidObjectID(), err
	}
	return b.Id, client.Seal(b.Id)
}

func EmptyBlobWriter(client *IPCClient) BlobWriter {
	meta := NewObjectMeta()
	meta.SetId(types.EmptyBlobID())
	meta.SetTypeName("vineyard::Blob")
	meta.AddKeyValue("length", 0)
	meta.AddKeyValue("nbytes", 0)
	meta.AddKeyValue("instance_id", client.InstanceID)
	meta.SetTransient(true)
	return BlobWriter{
		Id:     types.EmptyBlobID(),
		Size:   0,
		Meta:   meta,
		Buffer: nil,
	}
}

type BufferSet struct {
	buffers map[types.ObjectID]*arrow.Buffer
}

func (s *BufferSet) EmplaceBufferId(id types.ObjectID) error {
	buffer, ok := s.buffers[id]
	if ok && buffer != nil {
		return common.Error(
			common.KInvalid,
			fmt.Sprintf(
				"Invalid internal state: the buffer shouldn't has been filled, id = %d",
				id,
			),
		)
	}
	s.buffers[id] = nil
	return nil
}

func (s *BufferSet) EmplaceBuffer(id types.ObjectID, buffer *arrow.Buffer) error {
	b, ok := s.buffers[id]
	if !ok {
		return common.Error(
			common.KInvalid,
			fmt.Sprintf(
				"Invalid internal state: the buffer id hasn't been registered, id = %d",
				id,
			),
		)
	}
	if b != nil {
		return common.Error(
			common.KInvalid,
			fmt.Sprintf(
				"Invalid internal state: the buffer shouldn't has been filled, id = %d",
				id,
			),
		)
	}
	s.buffers[id] = buffer
	return nil
}

func (s *BufferSet) Reset() {
	if s.buffers == nil {
		s.buffers = make(map[types.ObjectID]*arrow.Buffer)
	}
	for k := range s.buffers {
		delete(s.buffers, k)
	}
}

func (s *BufferSet) GetBuffers() map[types.ObjectID]*arrow.Buffer {
	return s.buffers
}

func (s *BufferSet) GetBufferIds() []types.ObjectID {
	ids := make([]types.ObjectID, 0, len(s.buffers))
	for id := range s.buffers {
		ids = append(ids, id)
	}
	return ids
}

func (s *BufferSet) Contains(id types.ObjectID) bool {
	_, ok := s.buffers[id]
	return ok
}

func (s *BufferSet) Get(id types.ObjectID) (buffer *arrow.Buffer, err error) {
	buffer, ok := s.buffers[id]
	if ok {
		return buffer, nil
	} else {
		return nil, common.Error(common.KInvalid, fmt.Sprintf("Invalid internal state: the buffer id hasn't been registered, id = %d", id))
	}
}
