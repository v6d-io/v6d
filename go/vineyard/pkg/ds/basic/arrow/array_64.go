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

type Int64Array struct {
	client.Object
	*array.Int64
	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.Blob
	nullBitmap client.Blob
}

func (arr *Int64Array) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
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
	arr.Int64 = array.NewInt64Data(
		array.NewData(
			&arrow.Int64Type{},
			int(arr.Length),
			[]*memory.Buffer{arr.nullBitmap.Buffer, arr.buffer.Buffer},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *Int64Array) GetArray() arrow.Array {
	return arr.Int64
}

type Int64Builder struct {
	*array.Int64Builder
	*array.Int64

	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.BlobWriter
	nullBitmap client.BlobWriter
}

func NewInt64Builder() *Int64Builder {
	arr := &Int64Builder{}
	arr.Int64Builder = array.NewInt64Builder(memory.DefaultAllocator)
	return arr
}

func NewInt64BuilderFromArray(array *array.Int64) *Int64Builder {
	arr := &Int64Builder{}
	arr.Int64 = array
	return arr
}

func (arr *Int64Builder) Build(client *client.IPCClient) (err error) {
	if arr.Int64Builder != nil {
		arr.Int64 = arr.Int64Builder.NewInt64Array()
		defer arr.Int64.Release()
	}

	arr.Length = uint64(arr.Int64.Len())
	arr.NullCount = uint64(arr.Int64.NullN())
	arr.Offset = uint64(arr.Int64.Offset())

	buffers := arr.Int64.Data().Buffers()
	if arr.buffer, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.buffer.Buffer.Bytes(), buffers[1].Bytes())

	if arr.Int64.NullN() > 0 {
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

func (arr *Int64Builder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::NumericArray<int64>")
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

type Uint64Array struct {
	client.Object
	*array.Uint64
	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.Blob
	nullBitmap client.Blob
}

func (arr *Uint64Array) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
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
	arr.Uint64 = array.NewUint64Data(
		array.NewData(
			&arrow.Uint64Type{},
			int(arr.Length),
			[]*memory.Buffer{arr.nullBitmap.Buffer, arr.buffer.Buffer},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *Uint64Array) GetArray() arrow.Array {
	return arr.Uint64
}

type Uint64Builder struct {
	*array.Uint64Builder
	*array.Uint64

	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.BlobWriter
	nullBitmap client.BlobWriter
}

func NewUint64Builder() *Int64Builder {
	arr := &Int64Builder{}
	arr.Int64Builder = array.NewInt64Builder(memory.DefaultAllocator)
	return arr
}

func NewUint64BuilderFromArray(array *array.Uint64) *Uint64Builder {
	arr := &Uint64Builder{}
	arr.Uint64 = array
	return arr
}

func (arr *Uint64Builder) Build(client *client.IPCClient) (err error) {
	if arr.Uint64Builder != nil {
		arr.Uint64 = arr.Uint64Builder.NewUint64Array()
		defer arr.Uint64.Release()
	}

	arr.Length = uint64(arr.Uint64.Len())
	arr.NullCount = uint64(arr.Uint64.NullN())
	arr.Offset = uint64(arr.Uint64.Offset())

	buffers := arr.Uint64.Data().Buffers()
	if arr.buffer, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.buffer.Buffer.Bytes(), buffers[1].Bytes())

	if arr.Uint64.NullN() > 0 {
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

func (arr *Uint64Builder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::NumericArray<uint64>")
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

type Float64Array struct {
	client.Object
	*array.Float64
	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.Blob
	nullBitmap client.Blob
}

func (arr *Float64Array) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
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
	arr.Float64 = array.NewFloat64Data(
		array.NewData(
			&arrow.Float64Type{},
			int(arr.Length),
			[]*memory.Buffer{arr.nullBitmap.Buffer, arr.buffer.Buffer},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *Float64Array) GetArray() arrow.Array {
	return arr.Float64
}

type Float64Builder struct {
	*array.Float64Builder
	*array.Float64

	Length     uint64
	NullCount  uint64
	Offset     uint64
	buffer     client.BlobWriter
	nullBitmap client.BlobWriter
}

func NewFloat64Builder() *Float64Builder {
	arr := &Float64Builder{}
	arr.Float64Builder = array.NewFloat64Builder(memory.DefaultAllocator)
	return arr
}

func NewFloat64BuilderFromArray(array *array.Float64) *Float64Builder {
	arr := &Float64Builder{}
	arr.Float64 = array
	return arr
}

func (arr *Float64Builder) Build(client *client.IPCClient) (err error) {
	if arr.Float64Builder != nil {
		arr.Float64 = arr.Float64Builder.NewFloat64Array()
		defer arr.Float64.Release()
	}

	arr.Length = uint64(arr.Float64.Len())
	arr.NullCount = uint64(arr.Float64.NullN())
	arr.Offset = uint64(arr.Float64.Offset())

	buffers := arr.Float64.Data().Buffers()
	if arr.buffer, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.buffer.Buffer.Bytes(), buffers[1].Bytes())

	if arr.Float64.NullN() > 0 {
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

func (arr *Float64Builder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::NumericArray<double>")
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
