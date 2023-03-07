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

type StringArray struct {
	client.Object
	*array.String
	Length        uint64
	NullCount     uint64
	Offset        uint64
	bufferData    client.Blob
	bufferOffsets client.Blob
	nullBitmap    client.Blob
}

func (arr *StringArray) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
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
	if err = meta.GetMember(c, "buffer_data_", &arr.bufferData); err != nil {
		return err
	}
	if err = meta.GetMember(c, "buffer_offsets_", &arr.bufferOffsets); err != nil {
		return err
	}
	if err = meta.GetMember(c, "null_bitmap_", &arr.nullBitmap); err != nil {
		return err
	}
	arr.String = array.NewStringData(
		array.NewData(
			&arrow.StringType{},
			int(arr.Length),
			[]*memory.Buffer{
				arr.nullBitmap.Buffer,
				arr.bufferData.Buffer,
				arr.bufferOffsets.Buffer,
			},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *StringArray) GetArray() arrow.Array {
	return arr.String
}

type StringBuilder struct {
	*array.StringBuilder
	*array.String

	Length        uint64
	NullCount     uint64
	Offset        uint64
	bufferData    client.BlobWriter
	bufferOffsets client.BlobWriter
	nullBitmap    client.BlobWriter
}

func NewStringBuilder() *StringBuilder {
	arr := &StringBuilder{}
	arr.StringBuilder = array.NewStringBuilder(memory.DefaultAllocator)
	return arr
}

func NewStringBuilderFromArray(array *array.String) *StringBuilder {
	arr := &StringBuilder{}
	arr.String = array
	return arr
}

func (arr *StringBuilder) Build(client *client.IPCClient) (err error) {
	if arr.StringBuilder != nil {
		arr.String = arr.StringBuilder.NewStringArray()
		defer arr.String.Release()
	}

	arr.Length = uint64(arr.String.Len())
	arr.NullCount = uint64(arr.String.NullN())
	arr.Offset = uint64(arr.String.Offset())

	buffers := arr.String.Data().Buffers()
	if arr.bufferData, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.bufferData.Buffer.Bytes(), buffers[1].Bytes())
	if arr.bufferOffsets, err = client.CreateBuffer(uint64(buffers[2].Len())); err != nil {
		return nil
	}
	copy(arr.bufferOffsets.Buffer.Bytes(), buffers[2].Bytes())

	if arr.String.NullN() > 0 {
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

func (arr *StringBuilder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::StringArray")
	meta.AddKeyValue("length_", arr.Length)
	meta.AddKeyValue("null_count_", arr.NullCount)
	meta.AddKeyValue("offset_", arr.Offset)
	meta.SetNBytes(arr.bufferData.Size + arr.bufferOffsets.Size + arr.nullBitmap.Size)

	if _, err := arr.bufferData.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_data_", arr.bufferData.Meta)
	if _, err := arr.bufferOffsets.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_offsets_", arr.bufferOffsets.Meta)
	if _, err := arr.nullBitmap.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("null_bitmap_", arr.nullBitmap.Meta)
	return c.CreateMetaData(meta)
}

type LargeStringArray struct {
	client.Object
	*array.LargeString
	Length        uint64
	NullCount     uint64
	Offset        uint64
	bufferData    client.Blob
	bufferOffsets client.Blob
	nullBitmap    client.Blob
}

func (arr *LargeStringArray) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
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
	if err = meta.GetMember(c, "buffer_data_", &arr.bufferData); err != nil {
		return err
	}
	if err = meta.GetMember(c, "buffer_offsets_", &arr.bufferOffsets); err != nil {
		return err
	}
	if err = meta.GetMember(c, "null_bitmap_", &arr.nullBitmap); err != nil {
		return err
	}
	arr.LargeString = array.NewLargeStringData(
		array.NewData(
			&arrow.LargeStringType{},
			int(arr.Length),
			[]*memory.Buffer{
				arr.nullBitmap.Buffer,
				arr.bufferData.Buffer,
				arr.bufferOffsets.Buffer,
			},
			[]arrow.ArrayData{},
			int(arr.NullCount),
			int(arr.Offset),
		),
	)
	return nil
}

func (arr *LargeStringArray) GetArray() arrow.Array {
	return arr.LargeString
}

type LargeStringBuilder struct {
	*array.LargeStringBuilder
	*array.LargeString

	Length        uint64
	NullCount     uint64
	Offset        uint64
	bufferData    client.BlobWriter
	bufferOffsets client.BlobWriter
	nullBitmap    client.BlobWriter
}

func NewLargeStringBuilder() *LargeStringBuilder {
	arr := &LargeStringBuilder{}
	arr.LargeStringBuilder = array.NewLargeStringBuilder(memory.DefaultAllocator)
	return arr
}

func NewLargeStringBuilderFromArray(array *array.LargeString) *LargeStringBuilder {
	arr := &LargeStringBuilder{}
	arr.LargeString = array
	return arr
}

func (arr *LargeStringBuilder) Build(client *client.IPCClient) (err error) {
	if arr.LargeStringBuilder != nil {
		arr.LargeString = arr.LargeStringBuilder.NewLargeStringArray()
		defer arr.LargeString.Release()
	}

	arr.Length = uint64(arr.LargeString.Len())
	arr.NullCount = uint64(arr.LargeString.NullN())
	arr.Offset = uint64(arr.LargeString.Offset())

	buffers := arr.LargeString.Data().Buffers()
	if arr.bufferData, err = client.CreateBuffer(uint64(buffers[1].Len())); err != nil {
		return nil
	}
	copy(arr.bufferData.Buffer.Bytes(), buffers[1].Bytes())
	if arr.bufferOffsets, err = client.CreateBuffer(uint64(buffers[2].Len())); err != nil {
		return nil
	}
	copy(arr.bufferOffsets.Buffer.Bytes(), buffers[2].Bytes())

	if arr.LargeString.NullN() > 0 {
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

func (arr *LargeStringBuilder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::LargeStringArray")
	meta.AddKeyValue("length_", arr.Length)
	meta.AddKeyValue("null_count_", arr.NullCount)
	meta.AddKeyValue("offset_", arr.Offset)
	meta.SetNBytes(arr.bufferData.Size + arr.bufferOffsets.Size + arr.nullBitmap.Size)

	if _, err := arr.bufferData.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_data_", arr.bufferData.Meta)
	if _, err := arr.bufferOffsets.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_offsets_", arr.bufferOffsets.Meta)
	if _, err := arr.nullBitmap.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("null_bitmap_", arr.nullBitmap.Meta)
	return c.CreateMetaData(meta)
}
