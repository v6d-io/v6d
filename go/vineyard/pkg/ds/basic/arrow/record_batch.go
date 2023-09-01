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
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/apache/arrow/go/v11/arrow"
	"github.com/apache/arrow/go/v11/arrow/array"
	"github.com/apache/arrow/go/v11/arrow/ipc"
	"github.com/apache/arrow/go/v11/arrow/memory"
	"github.com/pkg/errors"
	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

func byteArrayToInt64Array(data []byte) []int64 {
	ret := make([]int64, len(data))
	for i, v := range data {
		ret[i] = int64(v)
	}
	return ret
}

func int64ArrayToByteArray(data []int64) []byte {
	ret := make([]byte, len(data))
	for i, v := range data {
		ret[i] = byte(v)
	}
	return ret
}

type Schema struct {
	client.Object
	*arrow.Schema
}

type SchemaBinaryWrapper struct {
	Bytes []int64 `json:"bytes"`
}

func (s *Schema) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if s.Id, err = meta.GetId(); err != nil {
		return err
	}
	s.Meta = meta
	schema, err := s.Meta.GetKeyValueString("schema_binary_")
	if err != nil {
		return err
	}
	fmt.Println("binary to construct ", schema)
	var binary SchemaBinaryWrapper
	err = json.Unmarshal([]byte(schema), &binary)
	if err != nil {
		return errors.New("schema_binary_ doesn't contain the bytes")
	}
	reader, err := ipc.NewReader(bytes.NewReader(int64ArrayToByteArray(binary.Bytes)))
	if err != nil {
		return err
	}
	s.Schema = reader.Schema()
	if reader.Err() != nil {
		return reader.Err()
	}
	return nil
}

type SchemaBuilder struct {
	*arrow.Schema
	binary []byte
}

func NewSchemaBuilder(schema *arrow.Schema) *SchemaBuilder {
	return &SchemaBuilder{
		Schema: schema,
	}
}

func (s *SchemaBuilder) Build(c *client.IPCClient) error {
	var buf bytes.Buffer
	writer := ipc.NewWriter(&buf, ipc.WithSchema(s.Schema))
	if err := writer.Close(); err != nil {
		return err
	}
	s.binary = buf.Bytes()
	return nil
}

func (s *SchemaBuilder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := s.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::SchemaProxy")
	fmt.Println("binary", s.binary)
	binary, err := json.Marshal(SchemaBinaryWrapper{Bytes: byteArrayToInt64Array(s.binary)})
	if err != nil {
		return types.InvalidObjectID(), err
	}
	fmt.Println("binary before add: ", byteArrayToInt64Array(s.binary))
	meta.AddKeyValue("schema_binary_", string(binary))
	meta.AddKeyValue("schema_textual_", s.Schema.String())
	return c.CreateMetaData(meta)
}

type RecordBatch struct {
	client.Object
	arrow.Record
}

func (r *RecordBatch) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if r.Id, err = meta.GetId(); err != nil {
		return err
	}
	r.Meta = meta
	var schema Schema
	if err = r.Meta.GetMember(c, "schema_", &schema); err != nil {
		return err
	}
	rowNum, err := r.Meta.GetKeyValueUint("row_num_")
	if err != nil {
		return err
	}
	columnSize, err := r.Meta.GetKeyValueInt("__columns_-size")
	if err != nil {
		return err
	}
	columns := make([]arrow.Array, 0, columnSize)
	for i := 0; i < columnSize; i++ {
		columnMeta, err := meta.GetMemberMeta(fmt.Sprintf("__columns_-%d", i))
		if err != nil {
			return err
		}
		column, err := ConstructArray(c, columnMeta)
		if err != nil {
			return err
		}
		columns = append(columns, column.GetArray())
	}
	r.Record = array.NewRecord(schema.Schema, columns, int64(rowNum))
	return nil
}

type RecordBuilder struct {
	*array.RecordBuilder
	arrow.Record

	Schema     *SchemaBuilder
	NumColumns uint64
	NumRows    uint64
	Columns    []types.ObjectID
}

func NewRecordBuilder(schema *arrow.Schema) *RecordBuilder {
	return &RecordBuilder{
		RecordBuilder: array.NewRecordBuilder(memory.DefaultAllocator, schema),
	}
}

func NewRecordBuilderFromRecord(record arrow.Record) *RecordBuilder {
	return &RecordBuilder{
		Record: record,
	}
}

func (r *RecordBuilder) Build(c *client.IPCClient) error {
	if r.RecordBuilder != nil {
		r.Record = r.RecordBuilder.NewRecord()
		defer r.Record.Release()
	}

	r.Schema = NewSchemaBuilder(r.Record.Schema())
	if err := r.Schema.Build(c); err != nil {
		return err
	}
	r.NumColumns = uint64(r.Record.NumCols())
	r.NumRows = uint64(r.Record.NumRows())
	r.Columns = make([]types.ObjectID, 0, r.NumColumns)
	for i := int64(0); i < r.Record.NumCols(); i++ {
		column := r.Record.Column(int(i))
		columnId, err := BuildArray(c, column)
		if err != nil {
			return err
		}
		r.Columns = append(r.Columns, columnId)
	}
	return nil
}

func (r *RecordBuilder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := r.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::RecordBatch")
	schemaId, err := r.Schema.Seal(c)
	if err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMemberId("schema_", schemaId)
	meta.AddKeyValue("column_num_", r.NumColumns)
	meta.AddKeyValue("row_num_", r.NumRows)
	meta.AddKeyValue("__columns_-size", r.NumColumns)
	for i, column := range r.Columns {
		meta.AddMemberId(fmt.Sprintf("__columns_-%d", i), column)
	}
	return c.CreateMetaData(meta)
}
