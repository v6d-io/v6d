/**
 * Copyright 2020-2022 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "client/client.h"
#include "common/util/env.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/logging.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/loader/arrow_fragment_loader.h"

namespace vineyard {

namespace detail {

struct loader_options {
  std::vector<std::string> efiles;
  std::vector<std::string> vfiles;
  bool directed = true;
  bool generate_eid = false;
  bool string_oid = false;
};

static inline bool parse_boolean_value(std::string const& value) {
  return value == "y" || value == "yes" || value == "t" || value == "true" ||
         value == "on" || value == "1";
}

static inline bool parse_boolean_value(const char* value) {
  return parse_boolean_value(std::string(value));
}

static inline bool parse_boolean_value(json const& value) {
  if (value.is_boolean()) {
    return value.get<bool>();
  } else if (value.is_number_integer()) {
    return value.get<int>() != 0;
  } else if (value.is_string()) {
    std::string const& s = value.get_ref<const std::string&>();
    return parse_boolean_value(s);
  } else {
    return false;
  }
}

static inline bool parse_options_from_args(struct loader_options& options,
                                           int current_index, int argc,
                                           char** argv) {
  int edge_label_num = atoi(argv[current_index++]);
  for (int index = 0; index < edge_label_num; ++index) {
    options.efiles.push_back(argv[current_index++]);
  }

  int vertex_label_num = atoi(argv[current_index++]);
  for (int index = 0; index < vertex_label_num; ++index) {
    options.vfiles.push_back(argv[current_index++]);
  }

  if (argc > current_index) {
    options.directed = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.generate_eid = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.string_oid = parse_boolean_value(argv[current_index++]);
  }
  return true;
}

static inline bool parse_options_from_config_json(
    struct loader_options& options, std::string const& config_json) {
  std::ifstream config_file(config_json);
  std::string config_json_content((std::istreambuf_iterator<char>(config_file)),
                                  std::istreambuf_iterator<char>());
  vineyard::json config = vineyard::json::parse(config_json_content);
  if (config.contains("vertices")) {
    for (auto const& item : config["vertices"]) {
      auto vfile = vineyard::ExpandEnvironmentVariables(
                       item["data_path"].get<std::string>()) +
                   "#label=" + item["label"].get<std::string>();
      if (item.contains("options")) {
        vfile += "#" + item["options"].get<std::string>();
      }
      options.vfiles.push_back(vfile);
    }
  }
  if (config.contains("edges")) {
    for (auto const& item : config["edges"]) {
      auto efile = vineyard::ExpandEnvironmentVariables(
                       item["data_path"].get<std::string>()) +
                   "#label=" + item["label"].get<std::string>() +
                   "#src_label=" + item["src_label"].get<std::string>() +
                   "#dst_label=" + item["dst_label"].get<std::string>();
      if (item.contains("options")) {
        efile += "#" + item["options"].get<std::string>();
      }
      options.efiles.push_back(efile);
    }
  }
  if (config.contains("directed")) {
    options.directed = parse_boolean_value(config["directed"]);
  }
  if (config.contains("generate_eid")) {
    options.generate_eid = parse_boolean_value(config["generate_eid"]);
  }
  if (config.contains("string_oid")) {
    options.string_oid = parse_boolean_value(config["string_oid"]);
  }
  return true;
}

}  // namespace detail

static void loading_vineyard_graph(
    struct detail::loader_options const& options) {
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  vineyard::Client& client = vineyard::Client::Default();

  MPI_Barrier(comm_spec.comm());
  vineyard::ObjectID fragment_group_id;
  if (options.string_oid) {
    auto loader = std::make_unique<vineyard::ArrowFragmentLoader<
        std::string, vineyard::property_graph_types::VID_TYPE>>(
        client, comm_spec, options.efiles, options.vfiles,
        options.directed != 0, options.generate_eid != 0);

    fragment_group_id = loader->LoadFragmentAsFragmentGroup().value();
  } else {
    auto loader = std::make_unique<vineyard::ArrowFragmentLoader<
        vineyard::property_graph_types::OID_TYPE,
        vineyard::property_graph_types::VID_TYPE>>(
        client, comm_spec, options.efiles, options.vfiles,
        options.directed != 0, options.generate_eid != 0);

    fragment_group_id = loader->LoadFragmentAsFragmentGroup().value();
  }

  LOG(INFO) << "[fragment group id]: " << fragment_group_id;
  MPI_Barrier(comm_spec.comm());

  std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(fragment_group_id));

  for (const auto& pair : fg->Fragments()) {
    LOG(INFO) << "[frag-" << pair.first << "]: " << pair.second;
  }
  MPI_Barrier(comm_spec.comm());
}

}  // namespace vineyard

int main(int argc, char** argv) {
  if (argc < 3) {
    printf(R"r(Usage: loading vertices and edges as vineyard graph.

    -     ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] \
                                   <e_label_num> <efiles...> <v_label_num> <vfiles...> \
                                   [directed] [generate_eid] [string_oid]

    - or: ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] --config <config.json>

          The config is a json file and should look like

          {
              "vertices": [
                  {
                      "data_path": "....",
                      "label": "...",
                      "options": "...."
                  },
                  ...
              ],
              "edges": [
                  {
                      "data_path": "",
                      "label": "",
                      "src_label": "",
                      "dst_label": "",
                      "options": ""
                  },
                  ...
              ],
              "directed": 1, # 0 or 1
              "generate_eid": 1, # 0 or 1
              "string_oid": 0 # 0 or 1
          })r");
    return 1;
  }

  std::string ipc_socket;
  struct vineyard::detail::loader_options options;

  int current_index = 1;
  if ((std::string(argv[current_index]) == "--socket") ||
      (std::string(argv[current_index]) == "-socket")) {
    current_index++;
    ipc_socket = argv[current_index++];
    LOG(INFO) << "Using vineyard IPC socket '" << ipc_socket << "'";
  } else {
    LOG(INFO) << "Resolve vineyard IPC socket from environment variable "
                 "'VINEYARD_IPC_SOCKET': '"
              << vineyard::read_env("VINEYARD_IPC_SOCKET") << "'";
  }
  if ((std::string(argv[current_index]) == "--config") ||
      (std::string(argv[current_index]) == "-config")) {
    current_index++;
    if (!vineyard::detail::parse_options_from_config_json(
            options, argv[current_index])) {
      exit(-1);
    }
  } else {
    if (!vineyard::detail::parse_options_from_args(options, current_index, argc,
                                                   argv)) {
      exit(-1);
    }
  }

  grape::InitMPIComm();
  { vineyard::loading_vineyard_graph(options); }
  grape::FinalizeMPIComm();

  return 0;
}
