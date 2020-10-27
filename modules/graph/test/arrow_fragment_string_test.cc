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

#include <stdio.h>

#include <fstream>
#include <string>

#include "glog/logging.h"

#include "grape/grape.h"
#include "grape/util.h"

#include "client/client.h"
#include "graph/fragment/arrow_fragment.h"
#include "graph/loader/arrow_fragment_loader.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using FragmentType = ArrowFragment<std::string, uint64_t>;
using ProjectedFragmentType =
    ArrowProjectedFragment<std::string, uint64_t, double, int64_t>;

std::string generate_path(const std::string& prefix, int part_num) {
  if (part_num == 1) {
    return prefix;
  } else {
    std::string ret;
    bool first = true;
    for (int i = 0; i < part_num; ++i) {
      if (first) {
        first = false;
        ret += (prefix + "_" + std::to_string(i));
      } else {
        ret += (";" + prefix + "_" + std::to_string(i));
      }
    }
    return ret;
  }
}

int main(int argc, char** argv) {
  if (argc < 8) {
    printf(
        "usage: ./run_string_oid <ipc_socket> "
        "<efile_prefix> <e_label_num> <efile_part>"
        "<vfile_prefix> <v_label_num> <vfile_part> [directed]\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  std::string epath = generate_path(argv[2], atoi(argv[4]));
  std::string vpath = generate_path(argv[5], atoi(argv[7]));
  int edge_label_num = atoi(argv[3]);
  int vertex_label_num = atoi(argv[6]);
  int directed = 1;
  if (argc >= 9) {
    directed = atoi(argv[8]);
  }

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  vineyard::ObjectID fragment_id = InvalidObjectID();
  {
    auto loader = std::make_unique<ArrowFragmentLoader<std::string, uint64_t>>(
        client, comm_spec, vertex_label_num, edge_label_num, epath, vpath,
        directed != 0);
    fragment_id = boost::leaf::try_handle_all(
        [&loader]() { return loader->LoadFragment(); },
        [](const GSError& e) {
          LOG(FATAL) << e.error_msg;
          return 0;
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(FATAL) << "Unmatched error " << unmatched;
          return 0;
        });
  }

  grape::FinalizeMPIComm();

  LOG(INFO) << "[worker-" << comm_spec.worker_id()
            << "] loaded graph to vineyard: " << VYObjectIDToString(fragment_id)
            << " ...";

  LOG(INFO) << "Passed arrow fragment with string oid test...";

  return 0;
}
