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

#include <stdio.h>

#include <fstream>
#include <string>

#include "client/client.h"
#include "common/util/logging.h"

#include "graph/grin/predefine.h"
#include "graph/grin/include/topology/structure.h"
#include "graph/grin/include/topology/vertexlist.h"
#include "graph/grin/include/topology/adjacentlist.h"
#include "graph/grin/include/partition/partition.h"
#include "graph/grin/include/partition/topology.h"
#include "graph/grin/include/partition/reference.h"
#include "graph/grin/include/property/type.h"
#include "graph/grin/include/property/topology.h"
#include "graph/grin/include/property/partition.h"
#include "graph/grin/include/property/propertylist.h"
#include "graph/grin/include/property/property.h"
#include "graph/grin/include/property/propertytable.h"

#include "graph/grin/src/predefine.h"
#include "graph/fragment/graph_schema.h"
#include "graph/loader/arrow_fragment_loader.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;

void sync_property(GRIN_PARTITIONED_GRAPH partitioned_graph, GRIN_PARTITION partition,
                   const char* edge_type_name, const char* vertex_property_name) {
  /*
    This example illustrates how to sync property values of vertices related to certain edge type.
    
    The input parameters are the partitioned_graph, the local partition,
    the edge_type_name (e.g., likes), the vertex_property_name (e.g., features)

    The task is to find all the destination vertices of "boundary edges" with type named "likes", and the vertices
    must have a property named "features". Here a boundary edge is an edge whose source vertex is a master vertex and
    the destination is a mirror vertex, given the context of "edge-cut" partition strategy that the underlying storage uses.
    Then for each of these vertices, we send the value of the "features" property to its master partition.
  */
  GRIN_GRAPH g = grin_get_local_graph_from_partition(partitioned_graph, partition);  // get local graph of partition
  auto _g = static_cast<GRIN_GRAPH_T*>(g);

  GRIN_EDGE_TYPE etype = grin_get_edge_type_by_name(g, edge_type_name);  // get edge type from name

  GRIN_VERTEX_TYPE_LIST src_vtypes = grin_get_src_types_from_edge_type(g, etype);  // get related source vertex type list
  GRIN_VERTEX_TYPE_LIST dst_vtypes = grin_get_dst_types_from_edge_type(g, etype);  // get related destination vertex type list

  size_t src_vtypes_num = grin_get_vertex_type_list_size(g, src_vtypes);
  size_t dst_vtypes_num = grin_get_vertex_type_list_size(g, dst_vtypes);

  for (size_t i = 0; i < src_vtypes_num; ++i) {  // iterate all pairs of src & dst vertex type
    GRIN_VERTEX_TYPE src_vtype = grin_get_vertex_type_from_list(g, src_vtypes, i);  // get src type
    GRIN_VERTEX_TYPE dst_vtype = grin_get_vertex_type_from_list(g, dst_vtypes, i);  // get dst type

    GRIN_VERTEX_PROPERTY dst_vp = grin_get_vertex_property_by_name(g, dst_vtype, vertex_property_name);  // get the property called "features" under dst type
    if (dst_vp == GRIN_NULL_VERTEX_PROPERTY) continue;  // filter out the pairs whose dst type does NOT have such a property called "features"
    
    GRIN_VERTEX_PROPERTY_TABLE dst_vpt = grin_get_vertex_property_table_by_type(g, dst_vtype);  // prepare property table of dst vertex type for later use
    GRIN_DATATYPE dst_vp_dt = grin_get_vertex_property_data_type(g, dst_vp); // prepare property type for later use

    GRIN_VERTEX_LIST __src_vl = grin_get_vertex_list(g);  // get the vertex list
    GRIN_VERTEX_LIST _src_vl = grin_filter_type_for_vertex_list(g, src_vtype, __src_vl);  // filter the vertex of source type
    GRIN_VERTEX_LIST src_vl = grin_filter_master_for_vertex_list(g, _src_vl);  // filter master vertices under source type
    
    size_t src_vl_num = grin_get_vertex_list_size(g, src_vl);

    for (size_t j = 0; j < src_vl_num; ++j) { // iterate the src vertex
      GRIN_VERTEX v = grin_get_vertex_from_list(g, src_vl, j);

#ifdef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
      GRIN_ADJACENT_LIST _adj_list = grin_get_adjacent_list(g, GRIN_DIRECTION::OUT, v);  // get the outgoing adjacent list of v
      GRIN_ADJACENT_LIST adj_list = grin_filter_edge_type_for_adjacent_list(g, etype, _adj_list);  // filter edges under etype
#else
      GRIN_ADJACENT_LIST adj_lsit = grin_get_adjacent_list(g, GRIN_DIRECTION::OUT, v);  // get the outgoing adjacent list of v
#endif

      size_t al_sz = grin_get_adjacent_list_size(g, adj_list);
      for (size_t k = 0; k < al_sz; ++k) {
#ifndef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
        GRIN_EDGE edge = grin_get_edge_from_adjacent_list(g, adj_list, k);
        GRIN_EDGE_TYPE edge_type = grin_get_edge_type(g, edge);
        if (!grin_equal_edge_type(g, edge_type, etype)) continue;
#endif
        GRIN_VERTEX u = grin_get_neighbor_from_adjacent_list(g, adj_list, k);  // get the dst vertex u
        const void* value = grin_get_value_from_vertex_property_table(g, dst_vpt, u, dst_vp);  // get the property value of "features" of u

        GRIN_VERTEX_REF uref = grin_get_vertex_ref_for_vertex(g, u);  // get the reference of u that can be recoginized by other partitions
        GRIN_PARTITION u_master_partition = grin_get_master_partition_from_vertex_ref(g, uref);  // get the master partition for u
        const char* uref_ser = grin_serialize_vertex_ref(g, uref);

        if (dst_vp_dt == GRIN_DATATYPE::Int64) {
          LOG(INFO) << "Message:" << uref_ser << " " << *(static_cast<const int64_t*>(value));
        }

        // send_value(u_master_partition, uref, dst_vp_dt, value);  // the value must be casted to the correct type based on dst_vp_dt before sending
      }
    }
  }
}


void Traverse(vineyard::Client& client, const grape::CommSpec& comm_spec,
              vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Loaded graph to vineyard: " << fragment_group_id;

  GRIN_PARTITIONED_GRAPH pg = get_partitioned_graph_by_object_id(client, fragment_group_id);

  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  size_t pnum = grin_get_partition_list_size(pg, local_partitions);
  LOG(INFO) << "Got partition num: " << pnum;

  // we only traverse the first partition for test
  GRIN_PARTITION partition = grin_get_partition_from_list(pg, local_partitions, 0);
  LOG(INFO) << "Partition: " << *(static_cast<unsigned*>(partition));
  sync_property(pg, partition, "knows", "value3");
}


namespace detail {

std::shared_ptr<arrow::ChunkedArray> makeInt64Array() {
  std::vector<int64_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  arrow::Int64Builder builder;
  CHECK_ARROW_ERROR(builder.AppendValues(data));
  std::shared_ptr<arrow::Array> out;
  CHECK_ARROW_ERROR(builder.Finish(&out));
  return arrow::ChunkedArray::Make({out}).ValueOrDie();
}

std::shared_ptr<arrow::Schema> attachMetadata(
    std::shared_ptr<arrow::Schema> schema, std::string const& key,
    std::string const& value) {
  std::shared_ptr<arrow::KeyValueMetadata> metadata;
  if (schema->HasMetadata()) {
    metadata = schema->metadata()->Copy();
  } else {
    metadata = std::make_shared<arrow::KeyValueMetadata>();
  }
  metadata->Append(key, value);
  return schema->WithMetadata(metadata);
}

std::vector<std::shared_ptr<arrow::Table>> makeVTables() {
  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          arrow::field("id", arrow::int64()),
          arrow::field("value1", arrow::int64()),
          arrow::field("value2", arrow::int64()),
          arrow::field("value3", arrow::int64()),
          arrow::field("value4", arrow::int64()),
      });
  schema = attachMetadata(schema, "label", "person");
  auto table = arrow::Table::Make(
      schema, {makeInt64Array(), makeInt64Array(), makeInt64Array(),
               makeInt64Array(), makeInt64Array()});
  return {table};
}

std::vector<std::vector<std::shared_ptr<arrow::Table>>> makeETables() {
  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          arrow::field("src_id", arrow::int64()),
          arrow::field("dst_id", arrow::int64()),
          arrow::field("value1", arrow::int64()),
          arrow::field("value2", arrow::int64()),
          arrow::field("value3", arrow::int64()),
          arrow::field("value4", arrow::int64()),
      });
  schema = attachMetadata(schema, "label", "knows");
  schema = attachMetadata(schema, "src_label", "person");
  schema = attachMetadata(schema, "dst_label", "person");
  auto table = arrow::Table::Make(
      schema, {makeInt64Array(), makeInt64Array(), makeInt64Array(),
               makeInt64Array(), makeInt64Array(), makeInt64Array()});
  return {{table}};
}
}  // namespace detail

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: ./arrow_fragment_test <ipc_socket> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  auto vtables = ::detail::makeVTables();
  auto etables = ::detail::makeETables();

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    {
      auto loader =
          std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                               property_graph_types::VID_TYPE>>(
              client, comm_spec, vtables, etables, directed != 0, true, true);
      vineyard::ObjectID fragment_group_id =
          loader->LoadFragmentAsFragmentGroup().value();
      Traverse(client, comm_spec, fragment_group_id);
    }
  }
  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment test...";

  return 0;
}
