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

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/graph_schema.h"
#include "graph/loader/arrow_fragment_loader.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;

void traverse_graph(std::shared_ptr<GraphType> graph, const std::string& path) {
  LabelType e_label_num = graph->edge_label_num();
  LabelType v_label_num = graph->edge_label_num();

  std::ofstream fout(path, std::ios::binary);
  for (LabelType v_label = 0; v_label != v_label_num; ++v_label) {
    auto iv = graph->InnerVertices(v_label);
    for (auto v : iv) {
      auto src_id = graph->GetId(v);
      for (LabelType e_label = 0; e_label != e_label_num; ++e_label) {
        auto oe = graph->GetOutgoingAdjList(v, e_label);
        for (auto& e : oe) {
          fout << src_id << " " << graph->GetId(e.neighbor()) << "\n";
        }
      }
    }
  }

  fout.flush();
  fout.close();
}

void WriteOut(vineyard::Client& client, const grape::CommSpec& comm_spec,
              vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Loaded graph to vineyard: " << fragment_group_id;
  std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(fragment_group_id));

  for (const auto& pair : fg->Fragments()) {
    LOG(INFO) << "[frag-" << pair.first << "]: " << pair.second;
  }

  // NB: only retrieve local fragments.
  auto locations = fg->FragmentLocations();
  for (const auto& pair : fg->Fragments()) {
    if (locations.at(pair.first) != client.instance_id()) {
      continue;
    }
    auto frag_id = pair.second;
    auto frag = std::dynamic_pointer_cast<GraphType>(client.GetObject(frag_id));
    auto schema = frag->schema();
    auto mg_schema = vineyard::MaxGraphSchema(schema);
    mg_schema.DumpToFile("/tmp/" + std::to_string(fragment_group_id) + ".json");

    LOG(INFO) << "[worker-" << comm_spec.worker_id()
              << "] loaded graph to vineyard: " << ObjectIDToString(frag_id)
              << " ...";
  }
}

int main(int argc, char** argv) {
  if (argc < 6) {
    printf(
        "usage: ./arrow_fragment_stream_test <ipc_socket> "
        "<e_stream_num> <estreams...> "
        "<v_stream_num> <vstreams...> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int edge_label_num = atoi(argv[index++]);
  std::vector<std::string> estreams;
  for (int i = 0; i < edge_label_num; ++i) {
    LOG(INFO) << "edges: " << ObjectIDToString(ObjectIDFromString(argv[index]));
    estreams.push_back("vineyard://" + std::string(argv[index++]));
  }

  int vertex_label_num = atoi(argv[index++]);
  std::vector<std::string> vstreams;
  for (int i = 0; i < vertex_label_num; ++i) {
    LOG(INFO) << "vertex: "
              << ObjectIDToString(ObjectIDFromString(argv[index]));
    vstreams.push_back("vineyard://" + std::string(argv[index++]));
  }

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  // Load from vstreams and estreams
  {
    auto loader =
        std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                             property_graph_types::VID_TYPE>>(
            client, comm_spec, vstreams, estreams, directed != 0);
    vineyard::ObjectID fragment_group_id = boost::leaf::try_handle_all(
        [&loader]() { return loader->LoadFragmentAsFragmentGroup(); },
        [](const GSError& e) {
          LOG(FATAL) << e.error_msg;
          return 0;
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(FATAL) << "Unmatched error " << unmatched;
          return 0;
        });
    WriteOut(client, comm_spec, fragment_group_id);
  }

  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment test (from stream)...";

  return 0;
}
