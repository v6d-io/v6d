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

using GraphType = vineyard::ArrowFragment<vineyard::property_graph_types::OID_TYPE,
                                          vineyard::property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;


void TraverseLocalGraph(void* partitioned_graph, void* partition) {
  vineyard::GRIN_ArrowFragment gaf;
  gaf.init(partitioned_graph, partition);
  auto g = gaf.get_graph();

  auto elabels = get_edge_type_list(g);
  auto e_label_num = get_edge_type_list_size(elabels);
  auto vlabels = get_vertex_type_list(g);
  auto v_label_num = get_vertex_type_list_size(vlabels);

  for (auto i = 0; i < e_label_num; ++i) {
    auto elabel = get_edge_type_from_list(elabels, i);
    auto props = get_edge_property_list_by_type(g, elabel);
    auto prop = get_edge_property_from_list(props, 0);
    auto prop_dt = get_edge_property_data_type(g, prop);
    auto dt_name = GetDataTypeName(prop_dt);
    for (auto j = 0; j < v_label_num; ++j) {
      auto vlabel = get_vertex_type_from_list(vlabels, j);
      auto iv = gaf.InnerVertices(vlabel);
      for (auto v: iv) {
        auto al = gaf.GetOutgoingAdjList(v, elabel);
        for (auto it: al) {
          auto neighbor = it.get_neighbor();
          auto edge = it.get_edge();
          if (dt_name == "double") {
            auto value = it.get_data<double>(prop);
          }
        }
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
  TraverseLocalGraph(pg, partition);
}

int main(int argc, char** argv) {
  if (argc < 6) {
    printf(
        "usage: ./grin_arrow_fragment_test <ipc_socket> <e_label_num> <efiles...> "
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
          std::make_unique<vineyard::ArrowFragmentLoader<vineyard::property_graph_types::OID_TYPE,
                                                         vineyard::property_graph_types::VID_TYPE>>(
              client, comm_spec, efiles, vfiles, directed != 0);
      vineyard::ObjectID fragment_group_id =
          loader->LoadFragmentAsFragmentGroup().value();
      traverse(client, comm_spec, fragment_group_id);
    }

    // Load from efiles
    {
      auto loader =
          std::make_unique<vineyard::ArrowFragmentLoader<vineyard::property_graph_types::OID_TYPE,
                                                         vineyard::property_graph_types::VID_TYPE>>(
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
