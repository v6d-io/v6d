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

package arrow

import (
	"github.com/apache/arrow/go/v11/arrow"
	"github.com/apache/arrow/go/v11/arrow/array"
	"github.com/apache/arrow/go/v11/arrow/memory"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type Int32Array struct {
	client.Object
	*array.Int32
	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.Blob
	nullBitmap client.Blob
}

func (arr *Int32Array) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if arr.Id, err = meta.GetId(); err != nil {
		return err
	}
	arr.Meta = meta
	if arr.Length, err = meta.GetKeyValueUInt64("length_"); err != nil {
		return err
	}
	if arr.NullCount, err = meta.GetKeyValueUInt64("null_count_"); err != nil {
		return err
	}
	if arr.Offset, err = meta.GetKeyValueUInt64("offset_"); err != nil {
		return err
	}
	if err = meta.GetMember(c, "buffer_", &arr.buffer); err != nil {
		return err
	}
	if err = meta.GetMember(c, "null_bitmap_", &arr.nullBitmap); err != nil {
		return err
	}
	arr.Int32 = array.NewInt32Data(
		array.NewData(
			&arrow.Int32Type{},
			int(arr.Length),
			[]*memory.Buffer{arr.nullBitmap.Buffer, arr.buffer.Buffer},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *Int32Array) GetArray() arrow.Array {
	return arr.Int32
}

type Int32Builder struct {
	*array.Int32Builder
	*array.Int32

	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.BlobWriter
	nullBitmap client.BlobWriter
}

func NewInt32Builder() *Int32Builder {
	arr := &Int32Builder{}
	arr.Int32Builder = array.NewInt32Builder(memory.DefaultAllocator)
	return arr
}

func NewInt32BuilderFromArray(array *array.Int32) *Int32Builder {
	arr := &Int32Builder{}
	arr.Int32 = array
	return arr
}

func (arr *Int32Builder) Build(client *client.IPCClient) (err error) {
	if arr.Int32Builder != nil {
		arr.Int32 = arr.Int32Builder.NewInt32Array()
		defer arr.Int32.Release()
	}

	arr.Length = uint64(arr.Int32.Len())
	arr.NullCount = uint64(arr.Int32.NullN())
	arr.Offset = uint64(arr.Int32.Offset())

	buffers := arr.Int32.Data().Buffers()
	if arr.buffer, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.buffer.Buffer.Bytes(), buffers[1].Bytes())

	if arr.Int32.NullN() > 0 {
		if arr.nullBitmap, err = client.CreateBuffer(uint64(buffers[0].Len())); err != nil {
			return nil
		}
		copy(arr.nullBitmap.Buffer.Bytes(), buffers[0].Bytes())
	} else {
		arr.nullBitmap, err = client.CreateBuffer(0)
		if err != nil {
			return err
		}
	}
	return nil
}

func (arr *Int32Builder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::NumericArray<int32>")
	meta.AddKeyValue("length_", arr.Length)
	meta.AddKeyValue("null_count_", arr.NullCount)
	meta.AddKeyValue("offset_", arr.Offset)
	meta.SetNBytes(arr.buffer.Size + arr.nullBitmap.Size)

	if _, err := arr.buffer.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_", arr.buffer.Meta)
	if _, err := arr.nullBitmap.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("null_bitmap_", arr.nullBitmap.Meta)
	return c.CreateMetaData(meta)
}

type Uint32Array struct {
	client.Object
	*array.Uint32
	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.Blob
	nullBitmap client.Blob
}

func (arr *Uint32Array) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if arr.Id, err = meta.GetId(); err != nil {
		return err
	}
	arr.Meta = meta
	if arr.Length, err = meta.GetKeyValueUInt64("length_"); err != nil {
		return err
	}
	if arr.NullCount, err = meta.GetKeyValueUInt64("null_count_"); err != nil {
		return err
	}
	if arr.Offset, err = meta.GetKeyValueUInt64("offset_"); err != nil {
		return err
	}
	if err = meta.GetMember(c, "buffer_", &arr.buffer); err != nil {
		return err
	}
	if err = meta.GetMember(c, "null_bitmap_", &arr.nullBitmap); err != nil {
		return err
	}
	arr.Uint32 = array.NewUint32Data(
		array.NewData(
			&arrow.Uint32Type{},
			int(arr.Length),
			[]*memory.Buffer{arr.nullBitmap.Buffer, arr.buffer.Buffer},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *Uint32Array) GetArray() arrow.Array {
	return arr.Uint32
}

type Uint32Builder struct {
	*array.Uint32Builder
	*array.Uint32

	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.BlobWriter
	nullBitmap client.BlobWriter
}

func NewUint32Builder() *Int32Builder {
	arr := &Int32Builder{}
	arr.Int32Builder = array.NewInt32Builder(memory.DefaultAllocator)
	return arr
}

func NewUint32BuilderFromArray(array *array.Uint32) *Uint32Builder {
	arr := &Uint32Builder{}
	arr.Uint32 = array
	return arr
}

func (arr *Uint32Builder) Build(client *client.IPCClient) (err error) {
	if arr.Uint32Builder != nil {
		arr.Uint32 = arr.Uint32Builder.NewUint32Array()
		defer arr.Uint32.Release()
	}

	arr.Length = uint64(arr.Uint32.Len())
	arr.NullCount = uint64(arr.Uint32.NullN())
	arr.Offset = uint64(arr.Uint32.Offset())

	buffers := arr.Uint32.Data().Buffers()
	if arr.buffer, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.buffer.Buffer.Bytes(), buffers[1].Bytes())

	if arr.Uint32.NullN() > 0 {
		if arr.nullBitmap, err = client.CreateBuffer(uint64(buffers[0].Len())); err != nil {
			return nil
		}
		copy(arr.nullBitmap.Buffer.Bytes(), buffers[0].Bytes())
	} else {
		arr.nullBitmap, err = client.CreateBuffer(0)
		if err != nil {
			return err
		}
	}
	return nil
}

func (arr *Uint32Builder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::NumericArray<uint32>")
	meta.AddKeyValue("length_", arr.Length)
	meta.AddKeyValue("null_count_", arr.NullCount)
	meta.AddKeyValue("offset_", arr.Offset)
	meta.SetNBytes(arr.buffer.Size + arr.nullBitmap.Size)

	if _, err := arr.buffer.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_", arr.buffer.Meta)
	if _, err := arr.nullBitmap.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("null_bitmap_", arr.nullBitmap.Meta)
	return c.CreateMetaData(meta)
}

type Float32Array struct {
	client.Object
	*array.Float32
	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.Blob
	nullBitmap client.Blob
}

func (arr *Float32Array) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if arr.Id, err = meta.GetId(); err != nil {
		return err
	}
	arr.Meta = meta
	if arr.Length, err = meta.GetKeyValueUInt64("length_"); err != nil {
		return err
	}
	if arr.NullCount, err = meta.GetKeyValueUInt64("null_count_"); err != nil {
		return err
	}
	if arr.Offset, err = meta.GetKeyValueUInt64("offset_"); err != nil {
		return err
	}
	if err = meta.GetMember(c, "buffer_", &arr.buffer); err != nil {
		return err
	}
	if err = meta.GetMember(c, "null_bitmap_", &arr.nullBitmap); err != nil {
		return err
	}
	arr.Float32 = array.NewFloat32Data(
		array.NewData(
			&arrow.Float32Type{},
			int(arr.Length),
			[]*memory.Buffer{arr.nullBitmap.Buffer, arr.buffer.Buffer},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *Float32Array) GetArray() arrow.Array {
	return arr.Float32
}

type Float32Builder struct {
	*array.Float32Builder
	*array.Float32

	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.BlobWriter
	nullBitmap client.BlobWriter
}

func NewFloat32Builder() *Float32Builder {
	arr := &Float32Builder{}
	arr.Float32Builder = array.NewFloat32Builder(memory.DefaultAllocator)
	return arr
}

func NewFloat32BuilderFromArray(array *array.Float32) *Float32Builder {
	arr := &Float32Builder{}
	arr.Float32 = array
	return arr
}

func (arr *Float32Builder) Build(client *client.IPCClient) (err error) {
	if arr.Float32Builder != nil {
		arr.Float32 = arr.Float32Builder.NewFloat32Array()
		defer arr.Float32.Release()
	}

	arr.Length = uint64(arr.Float32.Len())
	arr.NullCount = uint64(arr.Float32.NullN())
	arr.Offset = uint64(arr.Float32.Offset())

	buffers := arr.Float32.Data().Buffers()
	if arr.buffer, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.buffer.Buffer.Bytes(), buffers[1].Bytes())

	if arr.Float32.NullN() > 0 {
		if arr.nullBitmap, err = client.CreateBuffer(uint64(buffers[0].Len())); err != nil {
			return nil
		}
		copy(arr.nullBitmap.Buffer.Bytes(), buffers[0].Bytes())
	} else {
		arr.nullBitmap, err = client.CreateBuffer(0)
		if err != nil {
			return err
		}
	}
	return nil
}

func (arr *Float32Builder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::NumericArray<float>")
	meta.AddKeyValue("length_", arr.Length)
	meta.AddKeyValue("null_count_", arr.NullCount)
	meta.AddKeyValue("offset_", arr.Offset)
	meta.SetNBytes(arr.buffer.Size + arr.nullBitmap.Size)

	if _, err := arr.buffer.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_", arr.buffer.Meta)
	if _, err := arr.nullBitmap.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("null_bitmap_", arr.nullBitmap.Meta)
	return c.CreateMetaData(meta)
}
