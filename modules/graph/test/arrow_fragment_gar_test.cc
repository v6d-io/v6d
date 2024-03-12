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
#include "graph/loader/arrow_fragment_loader.h"
#include "graph/loader/gar_fragment_loader.h"
#include "graph/writer/arrow_fragment_writer.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using StringGraphType =
    ArrowFragment<std::string, property_graph_types::VID_TYPE>;
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
    const grape::CommSpec& comm_spec, std::shared_ptr<StringGraphType> graph,
    const std::string& output_path, const std::string& file_type) {
  auto writer = std::make_unique<ArrowFragmentWriter<StringGraphType>>();
  BOOST_LEAF_CHECK(writer->Init(
      graph, comm_spec, /* graph_name */ "graph", output_path,
      /* vertex_chunk_size */ 512,
      /* edge_chunk_size */ 1024, file_type,
      /* selected_vertices */ std::vector<std::string>{},
      /* selected_edges */ std::vector<std::string>{},
      /* selected_vertex_properties */
      std::unordered_map<std::string, std::vector<std::string>>{},
      /* selected_edge_properties */
      std::unordered_map<std::string, std::vector<std::string>>{}));
  BOOST_LEAF_CHECK(writer->WriteGraphInfo(output_path));
  BOOST_LEAF_CHECK(writer->WriteFragment());
  LOG(INFO) << "[worker-" << comm_spec.worker_id() << "] generate GAR files...";
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf(
        "usage: ./arrow_fragment_gar_test <ipc_socket> vdata_path edata_path "
        "output_path file_type\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  std::string v_file_path = vineyard::ExpandEnvironmentVariables(argv[index++]);
  std::string e_file_path = vineyard::ExpandEnvironmentVariables(argv[index++]);
  std::string output_path = vineyard::ExpandEnvironmentVariables(argv[index++]);
  std::string file_type = std::string(argv[index++]);

  std::string v_file_suffix = ".csv#header_row=true&label=person";
  std::string e_file_suffix =
      ".csv#header_row=true&label=knows&src_label=person&dst_label=person";

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    // Load graph from csv
    vineyard::ObjectID frag_group;
    {
      std::string vfile = v_file_path + v_file_suffix;
      std::string efile = e_file_path + e_file_suffix;
      auto loader = std::make_unique<
          ArrowFragmentLoader<std::string, property_graph_types::VID_TYPE>>(
          client, comm_spec, std::vector<std::string>{efile},
          std::vector<std::string>{vfile}, /*directed*/ true,
          /*generate_eid*/ false, /*retain_oid*/ true);
      frag_group = loader->LoadFragmentAsFragmentGroup().value();
      LOG(INFO) << "Loaded fragment group: " << ObjectIDToString(frag_group);
    }

    // Write out to GAR files
    {
      auto fg = std::dynamic_pointer_cast<ArrowFragmentGroup>(
          client.GetObject(frag_group));
      auto fid = comm_spec.WorkerToFrag(comm_spec.worker_id());
      auto frag_id = fg->Fragments().at(fid);
      auto arrow_frag =
          std::static_pointer_cast<StringGraphType>(client.GetObject(frag_id));
      write_out_to_gar(comm_spec, arrow_frag, output_path, file_type);
    }

    // Load from GAR files
    {
      std::string graph_yaml_path = output_path + "graph.graph.yaml";
      auto loader =
          std::make_unique<GARFragmentLoader<property_graph_types::OID_TYPE,
                                             property_graph_types::VID_TYPE>>(
              client, comm_spec);
      loader->Init(graph_yaml_path, /*selected_vertices*/ {},
                   /*selected_edges*/ {}, /*directed*/ true,
                   /*generate_eid*/ false,
                   /*store_in_local*/ false);
      vineyard::ObjectID fragment_id = loader->LoadFragment().value();

      std::shared_ptr<GraphType> graph =
          std::dynamic_pointer_cast<GraphType>(client.GetObject(fragment_id));
      LOG(INFO) << "[frag-" << graph->fid()
                << "]: " << ObjectIDToString(fragment_id);
      traverse_graph(graph,
                     "./xx/output_graph_" + std::to_string(graph->fid()));
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
