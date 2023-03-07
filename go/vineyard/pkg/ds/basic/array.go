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

package basic

import (
	"fmt"
	"unsafe"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/memory"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type Array[T types.Number] struct {
	client.Object
	Size uint64
	blob client.Blob

	Values []T
}

func (arr *Array[T]) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if arr.Id, err = meta.GetId(); err != nil {
		return err
	}
	arr.Meta = meta
	if arr.Size, err = meta.GetKeyValueUInt64("size_"); err != nil {
		return err
	}
	if err = meta.GetMember(c, "buffer_", &arr.blob); err != nil {
		return err
	}
	pointer, err := arr.blob.Pointer()
	if err != nil {
		return err
	}
	arr.Values = memory.CastFrom[T](pointer, arr.Size)
	return nil
}

func (arr *Array[T]) Len() uint64 {
	return arr.Size
}

func (arr *Array[T]) At(index uint64) T {
	return arr.Values[index]
}

type ArrayBuilder[T types.Number] struct {
	Size       uint64
	blobWriter client.BlobWriter

	Values []T
}

func NewArrayBuilder[T types.Number](
	c *client.IPCClient,
	size uint64,
) (arr *ArrayBuilder[T], err error) {
	arr = &ArrayBuilder[T]{Size: size}
	arr.blobWriter, err = c.CreateBuffer(size * (uint64)(unsafe.Sizeof(T(0))))
	if err != nil {
		return nil, err
	}
	pointer, err := arr.blobWriter.Pointer()
	if err != nil {
		return nil, err
	}
	arr.Values = unsafe.Slice((*T)(pointer), size)
	return arr, nil
}

func (arr *ArrayBuilder[T]) Build(client *client.IPCClient) error {
	return nil
}

func (arr *ArrayBuilder[T]) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := arr.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName(fmt.Sprintf("vineyard::Array<%T>", T(0)))
	meta.AddKeyValue("size_", arr.Size)
	meta.SetNBytes(arr.Size * (uint64)(unsafe.Sizeof(T(0))))

	if _, err := arr.blobWriter.Seal(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMember("buffer_", arr.blobWriter.Meta)
	return c.CreateMetaData(meta)
}

func (arr *ArrayBuilder[T]) At(index uint64, value T) {
	arr.Values[index] = value
}
