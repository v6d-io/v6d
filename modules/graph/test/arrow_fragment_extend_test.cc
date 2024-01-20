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

#include <algorithm>
#include <fstream>
#include <string>

#include "client/client.h"

#include "common/util/functions.h"
#include "common/util/uuid.h"
#include "graph/loader/arrow_fragment_loader.h"
#include "graph/loader/fragment_loader_utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<std::string, property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;

bool Validate(vineyard::Client& client, const grape::CommSpec& comm_spec,
              vineyard::ObjectID true_frag_group,
              vineyard::ObjectID test_frag_group) {
  std::shared_ptr<vineyard::ArrowFragmentGroup> test_fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(test_frag_group));
  std::shared_ptr<vineyard::ArrowFragmentGroup> true_fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(true_frag_group));
  auto edges = [](const std::shared_ptr<GraphType>& frag,
                  std::vector<std::tuple<std::string, std::string, int>>& oe,
                  std::vector<std::tuple<std::string, std::string, int>>& ie) {
    for (LabelType elabel = 0; elabel < frag->edge_label_num(); ++elabel) {
      for (LabelType vlabel = 0; vlabel < frag->vertex_label_num(); ++vlabel) {
        auto iv = frag->InnerVertices(vlabel);
        for (auto v : iv) {
          std::string src_id = frag->GetId(v);
          auto oes = frag->GetOutgoingAdjList(v, elabel);
          for (auto e : oes) {
            auto dst_id = frag->GetId(e.neighbor());
            int data = e.get_data<int64_t>(0);
            oe.emplace_back(src_id, dst_id, data);
          }
          if (frag->directed()) {
            auto ies = frag->GetIncomingAdjList(v, elabel);
            for (auto& e : ies) {
              auto dst_id = frag->GetId(e.neighbor());
              int data = e.get_int(0);
              ie.emplace_back(dst_id, src_id, data);
            }
          }
        }
      }
    }
    sort(oe.begin(), oe.end());
    sort(ie.begin(), ie.end());
  };

  auto inner_vertices = [](const std::shared_ptr<GraphType>& frag,
                           std::vector<std::string>& verts) {
    for (LabelType vlabel = 0; vlabel < frag->vertex_label_num(); ++vlabel) {
      auto iv = frag->InnerVertices(vlabel);
      for (auto v : iv) {
        std::string t = frag->GetId(v);
        verts.push_back(t);
      }
    }
    sort(verts.begin(), verts.end());
  };

  // NB: only retrieve local fragments.
  int f1_num = true_fg->total_frag_num(), f2_num = test_fg->total_frag_num();
  int idx1 = 0, idx2 = 0;
  std::vector<std::vector<std::string>> true_ivs(f1_num), test_ivs(f2_num);
  std::vector<std::vector<std::tuple<std::string, std::string, int>>> true_ies(
      f1_num),
      true_oes(f1_num), test_ies(f2_num), test_oes(f2_num);
  auto const& true_fragments = true_fg->Fragments();
  for (const auto& pair : true_fg->FragmentLocations()) {
    if (pair.second == client.instance_id()) {
      auto frag = std::dynamic_pointer_cast<GraphType>(
          client.GetObject(true_fragments.at(pair.first)));
      edges(frag, true_ies[idx1], true_oes[idx1]);
      inner_vertices(frag, true_ivs[idx1]);
      idx1++;
    }
  }

  auto const& test_fragments = test_fg->Fragments();
  for (const auto& pair : test_fg->FragmentLocations()) {
    if (pair.second == client.instance_id()) {
      auto frag = std::dynamic_pointer_cast<GraphType>(
          client.GetObject(test_fragments.at(pair.first)));
      inner_vertices(frag, test_ivs[idx2]);
      edges(frag, test_ies[idx2], test_oes[idx2]);
      idx2++;
    }
  }

  if (idx1 != idx2) {
    LOG(ERROR) << "fragment number is different";
    return false;
  }

  int len = idx1;
  for (int i = 0; i < len; ++i) {
    if (true_ivs[i].size() != test_ivs[i].size()) {
      LOG(ERROR) << "different inner vertices number";
      return false;
    }
    if (true_ies[i].size() != test_ies[i].size()) {
      LOG(ERROR) << "different inner edges number";
      return false;
    }
    if (true_oes[i].size() != test_oes[i].size()) {
      LOG(ERROR) << "different inner edges number";
      return false;
    }
    for (size_t j = 0; j < true_ivs[i].size(); ++j) {
      if (true_ivs[i][j] != test_ivs[i][j]) {
        LOG(ERROR) << "ground-truth v is " << true_ivs[i][j]
                   << "and program is " << test_ivs[i][j];
        return false;
      }
    }
    for (size_t j = 0; j < true_ies[i].size(); ++j) {
      if (true_ies[i][j] != test_ies[i][j]) {
        LOG(ERROR) << "different inner edge";
        return false;
      }
    }
    for (size_t j = 0; j < true_oes[i].size(); ++j) {
      if (true_oes[i][j] != test_oes[i][j]) {
        LOG(ERROR) << "different outgoing edge";
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf(
        "usage: ./arrow_fragment_label_data_extend <ipc_socket> [vdata_path] "
        "[edata_path]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);
  std::string v_file_path = vineyard::ExpandEnvironmentVariables(argv[index++]);
  std::string e_file_path = vineyard::ExpandEnvironmentVariables(argv[index++]);

  std::string v_file_suffix = ".csv#header_row=true&label=person";
  std::string e_file_suffix =
      ".csv#header_row=true&label=knows&src_label=person&dst_label=person";

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  using loader_t =
      ArrowFragmentLoader<std::string, property_graph_types::VID_TYPE>;

  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);
    vineyard::ObjectID true_frag_group;

    // first construct a basic graph
    {
      MPI_Barrier(comm_spec.comm());
      std::string vfile = v_file_path + v_file_suffix;
      std::string efile = e_file_path + e_file_suffix;
      auto loader = std::make_unique<loader_t>(
          client, comm_spec, std::vector<std::string>{efile},
          std::vector<std::string>{vfile}, /* directed */ 1,
          /*generate_eid*/ false);
      true_frag_group = loader->LoadFragmentAsFragmentGroup().value();
    }

    vineyard::ObjectID test_frag_group;
    LOG(INFO) << "start test extending";
    {
      MPI_Barrier(comm_spec.comm());
      // vertex_extending
      {
        std::string vfile = v_file_path + "_1" + v_file_suffix;
        auto loader = std::make_unique<loader_t>(
            client, comm_spec, std::vector<std::string>{},
            std::vector<std::string>{vfile}, /* directed */ 1,
            /*generate_eid*/ false, /* retain_oid */ true);
        test_frag_group = loader->LoadFragment().value();
        for (int i = 2; i < 4; ++i) {
          LOG(INFO) << "start extending " << i << "th vertex table";
          vfile = v_file_path + "_" + std::to_string(i) + v_file_suffix;
          auto loader = std::make_unique<loader_t>(
              client, comm_spec, std::vector<std::string>{},
              std::vector<std::string>{vfile}, /* directed */ 1,
              /*generate_eid*/ false, /* retain_oid */ true);
          test_frag_group =
              loader->AddDataToExistedVLabel(test_frag_group, 0).value();
          LOG(INFO) << "end extending " << i << "th vertex table";
        }
      }
      // edge_extending
      LOG(INFO) << "start edge extending";
      {
        std::string efile = e_file_path + "_1" + e_file_suffix;
        // first load one part e_file
        auto loader = std::make_unique<loader_t>(
            client, comm_spec, std::vector<std::string>{efile},
            std::vector<std::string>{}, /* directed */ 1,
            /*generate_eid*/ false, /* retain_oid */ true);
        test_frag_group = loader->AddLabelsToFragment(test_frag_group).value();
        for (int i = 2; i < 4; ++i) {
          LOG(INFO) << "start extending " << i << "th edge table";
          efile = e_file_path + "_" + std::to_string(i) + e_file_suffix;
          auto loader = std::make_unique<loader_t>(
              client, comm_spec, std::vector<std::string>{efile},
              std::vector<std::string>{}, /* directed */ 1,
              /*generate_eid*/ false, /* retain_oid */ true);
          test_frag_group =
              loader->AddDataToExistedELabel(test_frag_group, 0).value();
          LOG(INFO) << "end extending " << i << "th edge table";
        }
        test_frag_group =
            ConstructFragmentGroup(client, test_frag_group, comm_spec).value();
      }

      LOG(INFO) << "end edge extending";
      MPI_Barrier(comm_spec.comm());
    }
    bool valid = Validate(client, comm_spec, true_frag_group, test_frag_group);
    if (valid) {
      LOG(INFO) << "Passed arrow fragment data extend test...";
    } else {
      LOG(ERROR) << "Failed arrow fragment data extend test...";
    }
  }
  grape::FinalizeMPIComm();

  return 0;
}
