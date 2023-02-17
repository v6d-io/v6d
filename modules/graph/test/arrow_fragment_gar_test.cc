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

#ifdef ENABLE_GAR

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/graph_schema.h"
#include "graph/loader/gar_fragment_loader.h"
#include "graph/writer/arrow_fragment_writer.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;

void traverse_graph(std::shared_ptr<GraphType> graph, const std::string& path) {
  LabelType e_label_num = graph->edge_label_num();
  LabelType v_label_num = graph->vertex_label_num();

  for (LabelType v_label = 0; v_label != v_label_num; ++v_label) {
    std::ofstream fout(path + "_v_" + std::to_string(v_label),
                       std::ios::binary);
    auto iv = graph->InnerVertices(v_label);
    for (auto v : iv) {
      auto id = graph->GetId(v);
      fout << id << std::endl;
    }
    fout.flush();
    fout.close();
  }
  for (LabelType e_label = 0; e_label != e_label_num; ++e_label) {
    std::ofstream fout(path + "_e_" + std::to_string(e_label),
                       std::ios::binary);
    for (LabelType v_label = 0; v_label != v_label_num; ++v_label) {
      auto iv = graph->InnerVertices(v_label);
      for (auto v : iv) {
        auto src_id = graph->GetId(v);
        auto oe = graph->GetOutgoingAdjList(v, e_label);
        for (auto& e : oe) {
          fout << src_id << " " << graph->GetId(e.neighbor()) << "\n";
        }
      }
    }
    fout.flush();
    fout.close();
  }
}

boost::leaf::result<int> write_out_to_gar(
    const grape::CommSpec& comm_spec, std::shared_ptr<GraphType> graph,
    const GraphArchive::GraphInfo& graph_info) {
  auto writer = std::make_unique<ArrowFragmentWriter<GraphType>>(
      graph, comm_spec, graph_info);
  BOOST_LEAF_CHECK(writer->WriteFragment());
  LOG(INFO) << "[worker-" << comm_spec.worker_id() << "] generate GAR files...";
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf(
        "usage: ./arrow_fragment_test <ipc_socket> <graph_yaml_path>"
        "[directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  std::string graph_yaml_path =
      vineyard::ExpandEnvironmentVariables(argv[index++]);
  auto maybe_graph_info = GraphArchive::GraphInfo::Load(graph_yaml_path);
  if (maybe_graph_info.has_error()) {
    LOG(FATAL) << "Error: " << maybe_graph_info.status().message();
  }
  auto graph_info = maybe_graph_info.value();

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

    // Load from GAR files
    {
      auto loader =
          std::make_unique<GARFragmentLoader<property_graph_types::OID_TYPE,
                                             property_graph_types::VID_TYPE>>(
              client, comm_spec, graph_info, directed != 0);
      vineyard::ObjectID fragment_id = loader->LoadFragment().value();

      std::shared_ptr<GraphType> graph =
          std::dynamic_pointer_cast<GraphType>(client.GetObject(fragment_id));
      LOG(INFO) << "[frag-" << graph->fid()
                << "]: " << ObjectIDToString(fragment_id);
      traverse_graph(graph,
                     "./xx/output_graph_" + std::to_string(graph->fid()));
      write_out_to_gar(comm_spec, graph, graph_info);
    }
  }
  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment gar test...";

  return 0;
}

#else

int main(int argc, char** argv) {
  LOG(INFO) << "Arrow fragment gar test is disabled...";
  return 0;
}

#endif  // ENABLE_GAR
