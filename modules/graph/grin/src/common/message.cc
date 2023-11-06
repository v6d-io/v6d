/** Copyright 2020 Alibaba Group Holding Limited.
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

#include <google/protobuf/util/json_util.h>

#include "graph/grin/src/predefine.h"
#include "common/message.h"
#include "partition/partition.h"
#include "property/type.h"
#include "property/propertylist.h"
#include "property/property.h"
#include "property/primarykey.h"
#include "graph.pb.h"

void grin_destroy_graph_schema_msg(const char* s) {
  delete[] s;
}

void _set_storage_data_type(grin::StorageDataType* sdt, GRIN_DATATYPE dt) {
  switch (dt) {
    case GRIN_DATATYPE::Undefined:
      sdt->set_datatype(grin::DataType::DT_ANY);
      break;
    case GRIN_DATATYPE::Int32:
      sdt->set_datatype(grin::DataType::DT_SIGNED_INT32);
      break;
    case GRIN_DATATYPE::UInt32:
      sdt->set_datatype(grin::DataType::DT_UNSIGNED_INT32);
      break;
    case GRIN_DATATYPE::Int64:
      sdt->set_datatype(grin::DataType::DT_SIGNED_INT64);
      break;
    case GRIN_DATATYPE::UInt64:
      sdt->set_datatype(grin::DataType::DT_UNSIGNED_INT64);
      break;
    case GRIN_DATATYPE::Float:
      sdt->set_datatype(grin::DataType::DT_FLOAT);
      break;
    case GRIN_DATATYPE::Double:
      sdt->set_datatype(grin::DataType::DT_DOUBLE);
      break;
    case GRIN_DATATYPE::String:
      sdt->set_datatype(grin::DataType::DT_STRING);
      break;
    case GRIN_DATATYPE::Date32:
      sdt->set_datatype(grin::DataType::DT_SIGNED_INT32);
      break;
    case GRIN_DATATYPE::Time32:
      sdt->set_datatype(grin::DataType::DT_SIGNED_INT32);
      break;
    case GRIN_DATATYPE::Timestamp64:
      sdt->set_datatype(grin::DataType::DT_SIGNED_INT64);
      break;
    case GRIN_DATATYPE::FloatArray:
      sdt->set_datatype(grin::DataType::DT_FLOAT_ARRAY);
      break;
  }
}

const char* grin_get_graph_schema_msg(const char* uri) {
  GRIN_PARTITIONED_GRAPH pg =
      grin_get_partitioned_graph_from_storage(uri);
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  GRIN_PARTITION partition =
      grin_get_partition_from_list(pg, local_partitions, 0);
  GRIN_GRAPH g = grin_get_local_graph_by_partition(pg, partition);
  grin_destroy_partition(pg, partition);
  grin_destroy_partition_list(pg, local_partitions);
  grin_destroy_partitioned_graph(pg);

  grin::Graph s;
  s.set_uri(uri);
  auto schema = s.mutable_schema();

  GRIN_VERTEX_TYPE_LIST vtl = grin_get_vertex_type_list(g);
  size_t vtl_sz = grin_get_vertex_type_list_size(g, vtl);

  for (size_t i = 0; i < vtl_sz; ++i) {
    GRIN_VERTEX_TYPE vt = grin_get_vertex_type_from_list(g, vtl, i);
    auto svt = schema->add_vertex_types();
    svt->set_type_id(i);
    svt->set_type_name(grin_get_vertex_type_name(g, vt));
    
    GRIN_VERTEX_PROPERTY_LIST vpl = grin_get_vertex_property_list_by_type(g, vt);
    size_t vpl_sz = grin_get_vertex_property_list_size(g, vpl);
    for (size_t j = 0; j < vpl_sz; ++j) {
      GRIN_VERTEX_PROPERTY vp = grin_get_vertex_property_from_list(g, vpl, j);
      auto svp = svt->add_properties();
      svp->set_property_id(j);
      svp->set_property_name(grin_get_vertex_property_name(g, vt, vp));
      auto svpdt = svp->mutable_property_type();
      _set_storage_data_type(svpdt, grin_get_vertex_property_datatype(g, vp));
      grin_destroy_vertex_property(g, vp);
    }
    grin_destroy_vertex_property_list(g, vpl);

    GRIN_VERTEX_PROPERTY_LIST pks = grin_get_primary_keys_by_vertex_type(g, vt);
    size_t pks_sz = grin_get_vertex_property_list_size(g, vpl);
    for (size_t j = 0; j < pks_sz; ++j) {
      GRIN_VERTEX_PROPERTY vp = grin_get_vertex_property_from_list(g, pks, j);
      svt->add_primary_key_ids(grin_get_vertex_property_id(g, vt, vp));
      grin_destroy_vertex_property(g, vp);
    }
    grin_destroy_vertex_property_list(g, pks);
    grin_destroy_vertex_type(g, vt);
  }
  grin_destroy_vertex_type_list(g, vtl);

  GRIN_EDGE_TYPE_LIST etl = grin_get_edge_type_list(g);
  size_t etl_sz = grin_get_edge_type_list_size(g, etl);

  for (size_t i = 0; i < etl_sz; ++i) {
    GRIN_EDGE_TYPE et = grin_get_edge_type_from_list(g, etl, i);
    auto set = schema->add_edge_types();
    set->set_type_id(i);
    set->set_type_name(grin_get_edge_type_name(g, et));
    
    GRIN_EDGE_PROPERTY_LIST epl = grin_get_edge_property_list_by_type(g, et);
    size_t epl_sz = grin_get_edge_property_list_size(g, epl);
    for (size_t j = 0; j < epl_sz; ++j) {
      GRIN_EDGE_PROPERTY ep = grin_get_edge_property_from_list(g, epl, j);
      auto sep = set->add_properties();
      sep->set_property_id(j);
      sep->set_property_name(grin_get_edge_property_name(g, et, ep));
      auto sepdt = sep->mutable_property_type();
      _set_storage_data_type(sepdt, grin_get_edge_property_datatype(g, ep));
      grin_destroy_edge_property(g, ep);
    }
    grin_destroy_edge_property_list(g, epl);

    GRIN_VERTEX_TYPE_LIST src_vtl = grin_get_src_types_by_edge_type(g, et);
    GRIN_VERTEX_TYPE_LIST dst_vtl = grin_get_dst_types_by_edge_type(g, et);
    size_t vtl_sz = grin_get_vertex_type_list_size(g, src_vtl);
    for (size_t j = 0; j < vtl_sz; ++j) {
      auto srel = set->add_vertex_type_pair_relations();
      GRIN_VERTEX_TYPE src_vt = grin_get_vertex_type_from_list(g, src_vtl, j);
      GRIN_VERTEX_TYPE dst_vt = grin_get_vertex_type_from_list(g, dst_vtl, j);
      srel->set_src_type_id(grin_get_vertex_type_id(g, src_vt));
      srel->set_dst_type_id(grin_get_vertex_type_id(g, dst_vt));
      grin_destroy_vertex_type(g, src_vt);
      grin_destroy_vertex_type(g, dst_vt);
    }
    grin_destroy_vertex_type_list(g, src_vtl);
    grin_destroy_vertex_type_list(g, dst_vtl);
    grin_destroy_edge_type(g, et);
  }
  grin_destroy_edge_type_list(g, etl);

  auto sp = s.mutable_partition();
  auto sps = sp->add_partition_strategies();
  auto spse = sps->mutable_edge_cut();
  auto spsed = spse->mutable_directed_cut_edge_placement_strategies();
  spsed->add_cut_edge_placement_strategies(grin::PartitionStrategy_EdgeCut_DirectedEdgePlacementStrategy_DEPS_TO_SRC);
  spsed->add_cut_edge_placement_strategies(grin::PartitionStrategy_EdgeCut_DirectedEdgePlacementStrategy_DEPS_TO_DST);

  sp->add_vertex_property_placement_strategies(grin::PPS_ON_MASTER);
  sp->add_edge_property_placement_strategies(grin::PPS_ON_MASTER);
  sp->add_edge_property_placement_strategies(grin::PPS_ON_MIRROR);

  sp->add_master_vertices_sparse_index_strategies(grin::SIS_CSR);
  sp->add_master_vertices_sparse_index_strategies(grin::SIS_CSC);

  std::string msg;
  google::protobuf::util::MessageToJsonString(s, &msg);
  
  int len = msg.length() + 1;
  char* out = new char[len];
  snprintf(out, len, "%s", msg.c_str());
  return out;
}
