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
	"fmt"

	"github.com/apache/arrow/go/v11/arrow"
	"github.com/apache/arrow/go/v11/arrow/array"
	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type Table struct {
	client.Object
	arrow.Table
}

func (r *Table) Construct(c *client.IPCClient, meta *client.ObjectMeta) (err error) {
	if r.Id, err = meta.GetId(); err != nil {
		return err
	}
	r.Meta = meta
	var schema Schema
	if err = r.Meta.GetMember(c, "schema_", &schema); err != nil {
		return err
	}
	batchNum, err := r.Meta.GetKeyValueUint("partitions_-size")
	if err != nil {
		return err
	}
	batches := make([]arrow.Record, 0, batchNum)
	for i := 0; i < int(batchNum); i++ {
		var batch RecordBatch
		err = meta.GetMember(c, fmt.Sprintf("partitions_-%d", i), &batch)
		if err != nil {
			return err
		}
		batches = append(batches, batch.Record)
	}
	r.Table = array.NewTableFromRecords(schema.Schema, batches)
	return nil
}

type TableBuilder struct {
	Schema        *arrow.Schema
	SchemaBuilder *SchemaBuilder
	NumColumns    uint64
	NumRows       uint64

	Records []arrow.Record
	Batches []types.ObjectID
}

func NewTableBuilder(table arrow.Table) *TableBuilder {
	if table.NumCols() == 0 {
		return NewTableBuilderFromBatches(table.Schema(), []arrow.Record{})
	}
	reader := array.NewTableReader(table, -1)
	batches := make([]arrow.Record, 0)
	for reader.Next() {
		batches = append(batches, reader.Record())
	}
	return NewTableBuilderFromBatches(table.Schema(), batches)
}

func NewTableBuilderFromBatches(schema *arrow.Schema, records []arrow.Record) *TableBuilder {
	return &TableBuilder{
		Schema:  schema,
		Records: records,
	}
}

func (r *TableBuilder) Build(c *client.IPCClient) error {
	r.SchemaBuilder = NewSchemaBuilder(r.Schema)
	if err := r.SchemaBuilder.Build(c); err != nil {
		return err
	}
	r.Batches = make([]types.ObjectID, 0, len(r.Records))
	for _, record := range r.Records {
		r.NumRows += uint64(record.NumRows())
		builder := NewRecordBuilderFromRecord(record)
		if err := builder.Build(c); err != nil {
			return err
		}
		batchId, err := builder.Seal(c)
		if err != nil {
			return err
		}
		r.Batches = append(r.Batches, batchId)
	}
	r.NumColumns = uint64(len(r.Batches))
	return nil
}

func (r *TableBuilder) Seal(c *client.IPCClient) (types.ObjectID, error) {
	if err := r.Build(c); err != nil {
		return types.InvalidObjectID(), err
	}
	meta := client.NewObjectMeta()
	meta.SetTypeName("vineyard::Table")
	schemaId, err := r.SchemaBuilder.Seal(c)
	if err != nil {
		return types.InvalidObjectID(), err
	}
	meta.AddMemberId("schema_", schemaId)
	meta.AddKeyValue("num_columns_", r.NumColumns)
	meta.AddKeyValue("num_rows_", r.NumRows)
	meta.AddKeyValue("batch_num_", uint64(len(r.Batches)))
	meta.AddKeyValue("batch_num_", uint64(len(r.Batches)))
	meta.AddKeyValue("partitions_-size", r.NumColumns)
	for i, batch := range r.Batches {
		meta.AddMemberId(fmt.Sprintf("partitions_-%d", i), batch)
	}
	return c.CreateMetaData(meta)
}
