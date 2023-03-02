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

#include "graph/fragment/arrow_fragment.grin.h"
#include "graph/fragment/graph_schema.h"
#include "graph/loader/arrow_fragment_loader.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;



void sync_property(void* partitioned_graph, void* partition,
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
  auto g = get_local_graph_from_partition(partitioned_graph, partition);  // get local graph of partition

  auto etype = get_edge_type_by_name(g, edge_type_name);  // get edge type from name
  auto src_vtypes = get_src_types_from_edge_type(g, etype);  // get related source vertex type list
  auto dst_vtypes = get_dst_types_from_edge_type(g, etype);  // get related destination vertex type list

  auto src_vtypes_num = get_vertex_type_list_size(src_vtypes);
  auto dst_vtypes_num = get_vertex_type_list_size(dst_vtypes);
  assert(src_vtypes_num == dst_vtypes_num);  // the src & dst vertex type lists must be aligned

  for (auto i = 0; i < src_vtypes_num; ++i) {  // iterate all pairs of src & dst vertex type
    auto src_vtype = get_vertex_type_from_list(src_vtypes, i);  // get src type
    auto dst_vtype = get_vertex_type_from_list(dst_vtypes, i);  // get dst type

    auto dst_vp = get_vertex_property_by_name(g, dst_vtype, vertex_property_name);  // get the property called "features" under dst type
    if (dst_vp == NULL_PROPERTY) continue;  // filter out the pairs whose dst type does NOT have such a property called "features"
    
    auto dst_vpt = get_vertex_property_table_by_type(g, dst_vtype);  // prepare property table of dst vertex type for later use
    auto dst_vp_dt = get_vertex_property_data_type(g, dst_vp); // prepare property type for later use

    auto src_vl = get_master_vertices_by_type(g, src_vtype);  // we only need master vertices under source type
    
    auto src_vl_num = get_vertex_list_size(src_vl);
    for (auto j = 0; j < src_vl_num; ++j) { // iterate the src vertex
      auto v = get_vertex_from_list(src_vl, j);
      auto adj_list = get_adjacent_list_by_edge_type(g, Direction::OUT, v, etype);  // get the adjacent list of v with edges under etype
      bool check_flag = false;
      if (adj_list == NULL_LIST) {  // NULL_LIST means the storage does NOT support getting adj_list by edge type, note that list with size 0 is NOT a NULL_LIST
        // Then we should scan the full adj list and filter edge type by ourselves.
        adj_list = get_adjacent_list(g, Direction::OUT, v);
        bool check_flag = true;
      }

      auto al_sz = get_adjacent_list_size(adj_list);
      for (auto k = 0; k < al_sz; ++k) {
        if (check_flag) {
          auto edge = get_edge_from_adjacent_list(adj_list, k);
          auto edge_type = get_edge_type(g, edge);
          if (!equal_edge_type(edge_type, etype)) continue;
        }
        auto u = get_neighbor_from_adjacent_list(adj_list, k);  // get the dst vertex u
        auto value = get_value_from_vertex_property_table(dst_vpt, u, dst_vp);  // get the property value of "features" of u

        auto uref = get_vertex_ref_for_vertex(g, u);  // get the reference of u that can be recoginized by other partitions
        auto u_master_partition = get_master_partition_from_vertex_ref(g, uref);  // get the master partition for u

        // send_value(u_master_partition, uref, dst_vp_dt, value);  // the value must be casted to the correct type based on dst_vp_dt before sending
      }
    }
  }
}


void traverse(vineyard::Client& client, const grape::CommSpec& comm_spec,
              vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Loaded graph to vineyard: " << fragment_group_id;

  auto pg = get_partitioned_graph_by_object_id(client, fragment_group_id);
  auto local_partitions = get_local_partition_list(pg);
  size_t pnum = get_partition_list_size(local_partitions);
  assert(pnum > 0);

  // we only traverse the first partition for test
  auto partition = get_partition_from_list(local_partitions, 0);
  sync_property(pg, partition, "likes", "features");
}

int main(int argc, char** argv) {
  if (argc < 6) {
    printf(
        "usage: ./grin_test <ipc_socket> <e_label_num> <efiles...> "
        "<v_label_num> <vfiles...> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int edge_label_num = atoi(argv[index++]);
  std::vector<std::string> efiles;
  for (int i = 0; i < edge_label_num; ++i) {
    efiles.push_back(argv[index++]);
  }

  int vertex_label_num = atoi(argv[index++]);
  std::vector<std::string> vfiles;
  for (int i = 0; i < vertex_label_num; ++i) {
    vfiles.push_back(argv[index++]);
  }

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    // Load from efiles and vfiles
#if 0
    vineyard::ObjectID fragment_id = InvalidObjectID();
    MPI_Barrier(MPI_COMM_WORLD);
    double t = -GetCurrentTime();
    {
#if 0
    auto loader =
        std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                             property_graph_types::VID_TYPE>>(
            client, comm_spec, efiles, vfiles, directed != 0);
#else
      vfiles.clear();
      auto loader =
          std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                               property_graph_types::VID_TYPE>>(
              client, comm_spec, efiles, directed != 0);
#endif
      fragment_id = loader->LoadFragment().value();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t += GetCurrentTime();
    if (comm_spec.fid() == 0) {
      LOG(INFO) << "loading time: " << t;
    }

    {
      std::shared_ptr<GraphType> graph =
          std::dynamic_pointer_cast<GraphType>(client.GetObject(fragment_id));
      LOG(INFO) << "[frag-" << graph->fid()
                << "]: " << ObjectIDToString(fragment_id);
      traverse_graph(graph,
                     "./xx/output_graph_" + std::to_string(graph->fid()));
    }
    // client.DelData(fragment_id, true, true);
#else
    {
      auto loader =
          std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                               property_graph_types::VID_TYPE>>(
              client, comm_spec, efiles, vfiles, directed != 0);
      vineyard::ObjectID fragment_group_id =
          loader->LoadFragmentAsFragmentGroup().value();
      traverse(client, comm_spec, fragment_group_id);
    }

    // Load from efiles
    {
      auto loader =
          std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                               property_graph_types::VID_TYPE>>(
              client, comm_spec, efiles, directed != 0);
      vineyard::ObjectID fragment_group_id =
          loader->LoadFragmentAsFragmentGroup().value();
      traverse(client, comm_spec, fragment_group_id);
    }
#endif
  }
  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment test...";

  return 0;
}
