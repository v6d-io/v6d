/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "glog/logging.h"

#include "client/client.h"

#include "graph/fragment/arrow_fragment.h"

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

int main(int argc, char** argv) {
  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  std::string ipc_socket = std::string(argv[1]);
  std::string object_id_str = std::string(argv[2 + comm_spec.fid()]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  ObjectID object_id = VYObjectIDFromString(object_id_str);

  std::shared_ptr<GraphType> graph =
      std::dynamic_pointer_cast<GraphType>(client.GetObject(object_id));
  std::cout << "loaded graph...";

  traverse_graph(graph, "./xx/output_graph_" + std::to_string(graph->fid()));

  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment test...";

  return 0;
}
