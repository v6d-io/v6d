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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/apache/arrow/go/v11/arrow"
	"github.com/apache/arrow/go/v11/arrow/array"
	"github.com/apache/arrow/go/v11/arrow/memory"
	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

func TestTable(t *testing.T) {
	client, err := client.NewIPCClient(client.GetDefaultIPCSocket())
	if err != nil {
		t.Fatalf("connect to ipc server failed: %+v", err)
	}
	defer client.Disconnect()

	pool := memory.NewGoAllocator()

	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "f0-i32", Type: arrow.PrimitiveTypes.Int32},
			{Name: "f1-f64", Type: arrow.PrimitiveTypes.Float64},
		},
		nil,
	)

	// create record batch
	var objectId types.ObjectID
	{

		b := array.NewRecordBuilder(pool, schema)
		defer b.Release()

		b.Field(0).(*array.Int32Builder).AppendValues(
			[]int32{1, 2, 3, 4, 5, 6}, nil)
		b.Field(0).(*array.Int32Builder).AppendValues(
			[]int32{7, 8, 9, 10}, []bool{true, true, false, true})
		b.Field(1).(*array.Float64Builder).AppendValues(
			[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, nil)
		record := b.NewRecord()
		table := array.NewTableFromRecords(schema, []arrow.Record{record})
		defer table.Release()

		builder := NewTableBuilder(table)
		objectId, err = builder.Seal(client)
		if err != nil {
			t.Fatalf("seal table failed: %+v", err)
		}
	}

	// get record batch
	var table Table
	{
		err := client.GetObject(objectId, &table)
		if err != nil {
			t.Fatalf("get table failed: %+v", err)
		}
	}

	// valid record batch content
	{
		// check schema
		schema := table.Schema()
		assert.Equal(t, schema.Field(0).Name, "f0-i32", "schema field name not match")
		assert.Equal(t, schema.Field(0).Type.ID(), arrow.INT32, "schema field type not match")
		assert.Equal(t, schema.Field(1).Name, "f1-f64", "schema field name not match")
		assert.Equal(t, schema.Field(1).Type.ID(), arrow.FLOAT64, "schema field type not match")

		// check value
		assert.Equal(t, table.NumRows(), int64(10), "recordbatch row number not match")
		assert.Equal(t, table.NumCols(), int64(2), "recordbatch column number not match")

		batches := make([]arrow.Record, 0)
		reader := array.NewTableReader(table.Table, -1)
		for reader.Next() {
			batches = append(batches, reader.Record())
		}

		recordBatch := batches[0]

		// check column 0
		column0 := recordBatch.Column(0).(*array.Int32)
		assert.Equal(t, column0.Len(), 10, "column length not match")
		assert.Equal(t, column0.DataType().ID(), arrow.INT32, "column data type not match")
		for i := 0; i < 6; i++ {
			assert.Equal(t, column0.Value(i), int32(i+1), "column value not match")
		}

		// check column 1
		column1 := recordBatch.Column(1).(*array.Float64)
		assert.Equal(t, column1.Len(), 10, "column length not match")
		assert.Equal(t, column1.DataType().ID(), arrow.FLOAT64, "column data type not match")
		for i := 0; i < 10; i++ {
			assert.Equal(t, column1.Value(i), float64(i+1), "column value not match")
		}
	}
}
