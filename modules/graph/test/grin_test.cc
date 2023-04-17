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
#include "graph/grin/include/proto/message.h"

#include "graph/fragment/graph_schema.h"
#include "graph/loader/arrow_fragment_loader.h"

#include "graph/grin/src/proto/gie_schema.pb.h"

#include "google/protobuf/util/json_util.h"


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
    GRIN_VERTEX_LIST _src_vl = grin_select_type_for_vertex_list(g, src_vtype, __src_vl);  // filter the vertex of source type
    GRIN_VERTEX_LIST src_vl = grin_select_master_for_vertex_list(g, _src_vl);  // filter master vertices under source type
    
    size_t src_vl_num = grin_get_vertex_list_size(g, src_vl);

    for (size_t j = 0; j < src_vl_num; ++j) { // iterate the src vertex
      GRIN_VERTEX v = grin_get_vertex_from_list(g, src_vl, j);

#ifdef GRIN_TRAIT_SELECT_EDGE_TYPE_FOR_ADJACENT_LIST
      GRIN_ADJACENT_LIST _adj_list = grin_get_adjacent_list(g, GRIN_DIRECTION::OUT, v);  // get the outgoing adjacent list of v
      GRIN_ADJACENT_LIST adj_list = grin_select_edge_type_for_adjacent_list(g, etype, _adj_list);  // filter edges under etype
#else
      GRIN_ADJACENT_LIST adj_lsit = grin_get_adjacent_list(g, GRIN_DIRECTION::OUT, v);  // get the outgoing adjacent list of v
#endif

      size_t al_sz = grin_get_adjacent_list_size(g, adj_list);
      for (size_t k = 0; k < al_sz; ++k) {
#ifndef GRIN_TRAIT_SELECT_EDGE_TYPE_FOR_ADJACENT_LIST
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


void Traverse(GRIN_PARTITIONED_GRAPH pg) {
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  size_t pnum = grin_get_partition_list_size(pg, local_partitions);
  size_t vnum = grin_get_total_vertex_num(pg);
  size_t enumber = grin_get_total_edge_num(pg);
  LOG(INFO) << "Got partition num: " << pnum << " vertex num: " << vnum << " edge num: " << enumber;

  // we only traverse the first partition for test
  GRIN_PARTITION partition = grin_get_partition_from_list(pg, local_partitions, 0);
  LOG(INFO) << "Partition: " << *(static_cast<unsigned*>(partition));
  sync_property(pg, partition, "knows", "age");
}

gie::DataType data_type_cast(GRIN_DATATYPE dt) {
  switch (dt) {
    case GRIN_DATATYPE::Undefined:
      return gie::DataType::DT_UNKNOWN;
    case GRIN_DATATYPE::Int32:
      return gie::DataType::DT_SIGNED_INT32;
    case GRIN_DATATYPE::UInt32:
      return gie::DataType::DT_UNSIGNED_INT32;
    case GRIN_DATATYPE::Int64:
      return gie::DataType::DT_SIGNED_INT64;
    case GRIN_DATATYPE::UInt64:
      return gie::DataType::DT_UNSIGNED_INT64;
    case GRIN_DATATYPE::Float:
      return gie::DataType::DT_FLOAT;
    case GRIN_DATATYPE::Double:
      return gie::DataType::DT_DOUBLE;
    case GRIN_DATATYPE::String:
      return gie::DataType::DT_STRING;
    case GRIN_DATATYPE::Date32:
      return gie::DataType::DT_DATE;
    case GRIN_DATATYPE::Date64:
      return gie::DataType::DT_TIME;
    default:
      return gie::DataType::DT_UNKNOWN;
  }
}

std::string Convert(GRIN_PARTITIONED_GRAPH pg) {
  gie::Schema schema;
  // partition strategy
  auto ps = schema.mutable_partition_strategy();
  ps->set_topology(gie::GraphTopologyPartitionStrategy::GPS_EDGE_CUT_FOLLOW_BOTH);
  auto gpps = ps->mutable_property();
  auto item = gpps->mutable_by_entity();
  item->set_vertex_property_partition_strategy(gie::PropertyPartitionStrategy::PPS_MASTER);
  item->set_edge_property_partition_strategy(gie::PropertyPartitionStrategy::PPS_MASTER_MIRROR);
  
  // statistics
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  auto stats = schema.mutable_statistics();
  stats->set_num_partitions(grin_get_partition_list_size(pg, local_partitions));
  stats->set_num_vertices(grin_get_total_vertex_num(pg));
  stats->set_num_edges(grin_get_total_edge_num(pg));

  GRIN_PARTITION partition = grin_get_partition_from_list(pg, local_partitions, 0);
  GRIN_GRAPH g = grin_get_local_graph_from_partition(pg, partition);

  // vertex type
  GRIN_VERTEX_TYPE_LIST vtl = grin_get_vertex_type_list(g);
  size_t sz = grin_get_vertex_type_list_size(g, vtl);
  for (size_t i = 0; i < sz; ++i) {
    GRIN_VERTEX_TYPE vt = grin_get_vertex_type_from_list(g, vtl, i);
    auto vtype = schema.add_vertex_types();

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
    vtype->set_id(i);
#endif

#ifdef GRIN_WITH_VERTEX_TYPE_NAME
    vtype->set_name(grin_get_vertex_type_name(g, vt));
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY
    GRIN_VERTEX_PROPERTY_LIST vptl = grin_get_vertex_property_list_by_type(g, vt);
    size_t vpt_sz = grin_get_vertex_property_list_size(g, vptl);
    for (size_t j = 0; j < vpt_sz; ++j) {
      GRIN_VERTEX_PROPERTY vpt = grin_get_vertex_property_from_list(g, vptl, j);
      auto vprop = vtype->add_properties();
  
  #ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
      vprop->set_id(j);
  #endif

  #ifdef GRIN_WITH_VERTEX_PROPERTY_NAME
      vprop->set_name(grin_get_vertex_property_name(g, vpt));
  #endif

      auto dt = data_type_cast(grin_get_vertex_property_data_type(g, vpt));
      vprop->set_type(dt);
    }
#endif
    vtype->add_primary_keys("id");
    vtype->set_total_num(grin_get_vertex_num_by_type(g, vt));
  }

  // edge type
  GRIN_EDGE_TYPE_LIST etl = grin_get_edge_type_list(g);
  sz = grin_get_edge_type_list_size(g, etl);
  for (size_t i = 0; i < sz; ++i) {
    GRIN_EDGE_TYPE et = grin_get_edge_type_from_list(g, etl, i);
    auto etype = schema.add_edge_types();

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
    etype->set_id(i);
#endif

#ifdef GRIN_WITH_EDGE_TYPE_NAME
    etype->set_name(grin_get_edge_type_name(g, et));
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY
    GRIN_EDGE_PROPERTY_LIST eptl = grin_get_edge_property_list_by_type(g, et);
    size_t ept_sz = grin_get_edge_property_list_size(g, eptl);
    for (size_t j = 1; j < ept_sz; ++j) {
      GRIN_EDGE_PROPERTY ept = grin_get_edge_property_from_list(g, eptl, j);
      auto eprop = etype->add_properties();
  
  #ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
      eprop->set_id(j - 1);
  #endif

  #ifdef GRIN_WITH_EDGE_PROPERTY_NAME
      eprop->set_name(grin_get_edge_property_name(g, ept));
  #endif

      auto dt = data_type_cast(grin_get_edge_property_data_type(g, ept));
      eprop->set_type(dt);
    }
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY
    auto src_vtypes = grin_get_src_types_from_edge_type(g, et);
    auto dst_vtypes = grin_get_dst_types_from_edge_type(g, et);
    auto pair_sz = grin_get_vertex_type_list_size(g, src_vtypes);
    for (size_t j = 0; j < pair_sz; ++j) {
      auto src_vt = grin_get_vertex_type_from_list(g, src_vtypes, j);
      auto dst_vt = grin_get_vertex_type_from_list(g, dst_vtypes, j);
      auto pair = etype->add_src_dst_pairs();
  #ifdef GRIN_WITH_VERTEX_TYPE_NAME
      pair->set_src_type(grin_get_vertex_type_name(g, src_vt));
      pair->set_dst_type(grin_get_vertex_type_name(g, dst_vt));
  #endif
    }
#endif

    etype->set_total_num(grin_get_edge_num_by_type(g, et));
  }

  std::string schema_def;
  google::protobuf::util::MessageToJsonString(schema, &schema_def);
  return schema_def;
}

int main(int argc, char** argv) {
  std::cout << grin_get_static_storage_feature_msg() << std::endl;

  if (argc < 3) {
    printf("usage: ./arrow_fragment_test <ipc_socket> <object_id>\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  vineyard::ObjectID obj_id = std::stoull(argv[2]);

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    LOG(INFO) << "Loaded graph to vineyard: " << obj_id;
    std::string fg_id_str = std::to_string(obj_id);
    char** argv = new char*[2];
    argv[0] = new char[ipc_socket.length() + 1];
    argv[1] = new char[fg_id_str.length() + 1];
    strcpy(argv[0], ipc_socket.c_str());
    strcpy(argv[1], fg_id_str.c_str());
    GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(2, argv);

    std::cout << Convert(pg) << std::endl;
 
    Traverse(pg);

    delete[] argv[0];
    delete[] argv[1];
    delete[] argv;
  }

  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed grin test...";

  return 0;
}
