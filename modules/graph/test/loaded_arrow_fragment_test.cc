/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include <sys/stat.h>

#include <stdio.h>

#include <fstream>
#include <string>

#include "client/client.h"
#include "common/util/logging.h"

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

static void _mkdir(const char* dir) {
  char tmp[256];
  char* p = NULL;
  size_t len;

  snprintf(tmp, sizeof(tmp), "%s", dir);
  len = strlen(tmp);
  if (tmp[len - 1] == '/') {
    tmp[len - 1] = 0;
  }
  for (p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      mkdir(tmp, S_IRWXU);
      *p = '/';
    }
  }
  mkdir(tmp, S_IRWXU);
}

int test_traverse_loaded_graph(int argc, const char** argv) {
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  if (argc < 3) {
    printf(
        "usage ./load_arrow_fragment_test <ipc_socket> <fragment id 0> "
        "[<fragment id 1>] ...\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  std::string object_id_str = std::string(argv[2 + comm_spec.fid()]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  ObjectID object_id = ObjectIDFromString(object_id_str);
  bool exists = false;
  if (!client.Exists(object_id, exists).ok() || !exists) {
    object_id = std::strtoll(object_id_str.c_str(), nullptr, 10);
  }

  std::shared_ptr<GraphType> graph =
      std::dynamic_pointer_cast<GraphType>(client.GetObject(object_id));
  LOG(INFO) << "loaded graph ...";

  _mkdir("./graph_dumpped");
  traverse_graph(
      graph, "./graph_dumpped/output_graph_" + std::to_string(graph->fid()));
  LOG(INFO) << "dumpped graph to ./graph_dumpped/... ...";

  LOG(INFO) << "Passed arrow fragment test...";
  return 0;
}

int main(int argc, const char** argv) {
  grape::InitMPIComm();

  { test_traverse_loaded_graph(argc, argv); }
  grape::FinalizeMPIComm();

  return 0;
}
