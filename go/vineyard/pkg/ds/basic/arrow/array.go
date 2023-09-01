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
	"github.com/pkg/errors"
	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type Array interface {
	client.IObject
	GetArray() arrow.Array
}

func ConstructArray(c *client.IPCClient, meta *client.ObjectMeta) (array Array, err error) {
	typename, err := meta.GetTypeName()
	if err != nil {
		return nil, err
	}
	switch typename {
	case "vineyard::NumericArray<int32>":
		array = &Int32Array{}
	case "vineyard::NumericArray<uint32>":
		array = &Uint32Array{}
	case "vineyard::NumericArray<float>":
		array = &Float32Array{}
	case "vineyard::NumericArray<int64>":
		array = &Int64Array{}
	case "vineyard::NumericArray<uint64>":
		array = &Uint64Array{}
	case "vineyard::NumericArray<double>":
		array = &Float64Array{}
	case "vineyard::StringArray":
		array = &StringArray{}
	case "vineyard::LargeStringArray":
		array = &LargeStringArray{}
	default:
		return nil, errors.Errorf("unsupported array type: %s", typename)
	}
	return array, array.Construct(c, meta)
}

func BuildArray(c *client.IPCClient, arr arrow.Array) (types.ObjectID, error) {
	var builder client.IObjectBuilder
	switch arr.DataType().ID() {
	case arrow.INT32:
		builder = NewInt32BuilderFromArray(arr.(*array.Int32))
	case arrow.UINT32:
		builder = NewUint32BuilderFromArray(arr.(*array.Uint32))
	case arrow.FLOAT32:
		builder = NewFloat32BuilderFromArray(arr.(*array.Float32))
	case arrow.INT64:
		builder = NewInt64BuilderFromArray(arr.(*array.Int64))
	case arrow.UINT64:
		builder = NewUint64BuilderFromArray(arr.(*array.Uint64))
	case arrow.FLOAT64:
		builder = NewFloat64BuilderFromArray(arr.(*array.Float64))
	case arrow.STRING:
		builder = NewStringBuilderFromArray(arr.(*array.String))
	case arrow.LARGE_STRING:
		builder = NewLargeStringBuilderFromArray(arr.(*array.LargeString))
	default:
		return types.InvalidObjectID(), errors.Errorf(
			"unsupported array type: %s",
			arr.DataType().Name(),
		)
	}
	return builder.Seal(c)
}
