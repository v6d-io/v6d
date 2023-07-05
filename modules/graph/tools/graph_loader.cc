/**
 * Copyright 2020-2023 Alibaba Group Holding Limited.
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
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "grape/worker/comm_spec.h"

#include "client/client.h"
#include "common/util/env.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/version.h"

#include "graph/fragment/arrow_fragment_base.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/graph_schema.h"
#include "graph/tools/graph_loader.h"

namespace vineyard {

template <typename OID_T, typename VID_T>
class ArrowVertexMap;

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMap;

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
class ArrowFragment;

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T>
class ArrowFragmentLoader;

namespace detail {

void dump_normalized_schemas(Client& client, grape::CommSpec& comm_spec,
                             const ObjectID fragment_group_id,
                             const bool print_normalized_schema) {
  if (comm_spec.local_id() != 0) {
    return;
  }

  std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(fragment_group_id));

  auto const& fragments = fg->Fragments();
  for (const auto& item : fg->FragmentLocations()) {
    if (item.second == client.instance_id()) {
      auto frag = std::dynamic_pointer_cast<vineyard::ArrowFragmentBase>(
          client.GetObject(fragments.at(item.first)));
      vineyard::MaxGraphSchema schema(frag->schema());
      if (print_normalized_schema) {
        json root;
        schema.ToJSON(root);
        std::clog << root.dump(2);
      }
      std::string ofile = "/tmp/" + std::to_string(fragment_group_id) + ".json";
      schema.DumpToFile(ofile);
      LOG(INFO) << "The schema json has been dumped to '" << ofile << "'";
      break;
    }
  }
}

void print_graph_memory_usage(Client& client, grape::CommSpec& comm_spec,
                              const ObjectID fragment_group_id,
                              const bool print_memory_usage) {
  if (comm_spec.local_id() != 0) {
    return;
  }

  std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(fragment_group_id));

  auto const& fragments = fg->Fragments();
  for (const auto& item : fg->FragmentLocations()) {
    if (item.second == client.instance_id()) {
      ObjectMeta meta;
      VINEYARD_CHECK_OK(client.GetMetaData(fragments.at(item.first), meta));
      ObjectMeta vertexmap = meta.GetMemberMeta("vm_ptr_");
      json usages;
      size_t memory_usage = meta.MemoryUsage(usages, true);
      LOG(INFO) << "[frag-" << item.first << "]: " << item.second
                << "\n\tuses memory: " << prettyprint_memory_size(memory_usage)
                << "\n\tvertex map: "
                << prettyprint_memory_size(vertexmap.MemoryUsage());
      if (print_memory_usage) {
        std::clog << usages.dump(2);
      }
    }
  }
}

static inline bool parse_boolean_value(std::string const& value) {
  return value == "y" || value == "yes" || value == "t" || value == "true" ||
         value == "on" || value == "1";
}

static inline std::string normalize_data_type(std::string const& datatype) {
  if (datatype == "s" || datatype == "str" || datatype == "string" ||
      datatype == "std::string") {
    return "string";
  }
  if (datatype == "i" || datatype == "i32" || datatype == "int" ||
      datatype == "int32" || datatype == "int32_t") {
    return "int32";
  }
  return "int64";  // default
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

static inline progressive_t parse_progressive_enum(const std::string& value) {
  if (value == "whole") {
    return progressive_t::WHOLE;
  }
  if (value == "step_by_step") {
    return progressive_t::STEP_BY_STEP;
  }
  if (true /* value == "none" */) {
    return progressive_t::NONE;
  }
}

static inline std::string string_join(std::vector<std::string> const& srcs,
                                      std::string const sep) {
  std::stringstream ss;
  if (!srcs.empty()) {
    ss << srcs[0];
    for (size_t i = 1; i < srcs.size(); ++i) {
      ss << sep << srcs[i];
    }
  }
  return ss.str();
}

static inline progressive_t parse_progressive_enum(const char* value) {
  return parse_progressive_enum(std::string(value));
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
    options.compact_edges = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.generate_eid = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.retain_oid = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.oid_type = normalize_data_type(argv[current_index++]);
  }
  if (argc > current_index) {
    options.large_vid = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.large_eid = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.local_vertex_map = parse_boolean_value(argv[current_index++]);
  }
  if (argc > current_index) {
    options.progressive = parse_progressive_enum(argv[current_index++]);
  }
  if (argc > current_index) {
    options.catch_leaf_errors = parse_boolean_value(argv[current_index++]);
  }
  return true;
}

static inline bool parse_options_from_config_json(
    struct loader_options& options, std::string const& config_json) {
  std::ifstream config_file(config_json);
  std::string config_json_content((std::istreambuf_iterator<char>(config_file)),
                                  std::istreambuf_iterator<char>());
#if NLOHMANN_JSON_VERSION_MAJOR > 3 || \
    (NLOHMANN_JSON_VERSION_MAJOR == 3 && NLOHMANN_JSON_VERSION_MINOR >= 9)
  vineyard::json config =
      vineyard::json::parse(config_json_content, nullptr, true, true);
#else
  vineyard::json config =
      vineyard::json::parse(config_json_content, nullptr, true);
#endif
  if (config.contains("vertices")) {
    std::map<std::string, std::vector<std::string>> vertices;
    for (auto const& item : config["vertices"]) {
      auto vfile = vineyard::ExpandEnvironmentVariables(
                       item["data_path"].get<std::string>()) +
                   "#label=" + item["label"].get<std::string>();
      if (item.contains("options")) {
        vfile += "#" + item["options"].get<std::string>();
      }
      vertices[item["label"].get<std::string>()].push_back(vfile);
      // options.vfiles.push_back(vfile);
    }
    for (auto const& item : vertices) {
      options.vfiles.push_back(string_join(item.second, ";"));
    }
  }
  if (config.contains("edges")) {
    std::map<std::string, std::vector<std::string>> edges;
    for (auto const& item : config["edges"]) {
      auto efile = vineyard::ExpandEnvironmentVariables(
                       item["data_path"].get<std::string>()) +
                   "#label=" + item["label"].get<std::string>() +
                   "#src_label=" + item["src_label"].get<std::string>() +
                   "#dst_label=" + item["dst_label"].get<std::string>();
      if (item.contains("options")) {
        efile += "#" + item["options"].get<std::string>();
      }
      edges[item["label"].get<std::string>()].push_back(efile);
      // options.efiles.push_back(efile);
    }
    for (auto const& item : edges) {
      options.efiles.push_back(string_join(item.second, ";"));
    }
  }
  if (config.contains("directed")) {
    options.directed = parse_boolean_value(config["directed"]);
  }
  if (config.contains("generate_eid")) {
    options.generate_eid = parse_boolean_value(config["generate_eid"]);
  }
  if (config.contains("retain_oid")) {
    options.retain_oid = parse_boolean_value(config["retain_oid"]);
  }
  if (config.contains("oid_type")) {
    options.oid_type =
        normalize_data_type(config["oid_type"].get<std::string>());
  }
  if (config.contains("large_vid")) {
    options.large_vid = parse_boolean_value(config["large_vid"]);
  }
  if (config.contains("large_eid")) {
    options.large_eid = parse_boolean_value(config["large_eid"]);
  }
  if (config.contains("local_vertex_map")) {
    options.local_vertex_map = parse_boolean_value(config["local_vertex_map"]);
  }
  if (config.contains("progressive")) {
    options.progressive =
        parse_progressive_enum(config["progressive"].get<std::string>());
  }
  if (config.contains("catch_leaf_errors")) {
    options.catch_leaf_errors =
        parse_boolean_value(config["catch_leaf_errors"]);
  }
  if (config.contains("dump")) {
    options.dump = config["dump"].get<std::string>();
  }
  if (config.contains("dump_dry_run_rounds")) {
    options.dump_dry_run_rounds = config["dump_dry_run_rounds"].get<size_t>();
  }
  if (config.contains("print_memory_usage")) {
    options.print_memory_usage =
        parse_boolean_value(config["print_memory_usage"]);
  }
  if (config.contains("print_normalized_schema")) {
    options.print_normalized_schema =
        parse_boolean_value(config["print_normalized_schema"]);
  }
  if (config.contains("compact_edges")) {
    options.compact_edges = parse_boolean_value(config["compact_edges"]);
  }
  if (config.contains("use_perfect_hash")) {
    options.use_perfect_hash = parse_boolean_value(config["use_perfect_hash"]);
  }
  return true;
}

}  // namespace detail

static void loading_vineyard_graph(
    struct detail::loader_options const& options) {
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(options.vineyard_ipc_socket));

  MPI_Barrier(comm_spec.comm());
  vineyard::ObjectID fragment_group_id = InvalidObjectID();
  if (options.large_vid) {
    if (options.oid_type == "string") {
      fragment_group_id =
          detail::load_graph<std::string, uint64_t>(client, comm_spec, options);
    } else if (options.oid_type == "int32") {
      fragment_group_id =
          detail::load_graph<int32_t, uint64_t>(client, comm_spec, options);
    } else {
      fragment_group_id =
          detail::load_graph<int64_t, uint64_t>(client, comm_spec, options);
    }
  } else {
    if (options.oid_type == "string") {
      fragment_group_id =
          detail::load_graph<std::string, uint32_t>(client, comm_spec, options);
    } else if (options.oid_type == "int32") {
      fragment_group_id =
          detail::load_graph<int32_t, uint32_t>(client, comm_spec, options);
    } else {
      fragment_group_id =
          detail::load_graph<int64_t, uint32_t>(client, comm_spec, options);
    }
  }

  MPI_Barrier(comm_spec.comm());
  LOG(INFO) << "[fragment group id]: " << fragment_group_id;
  detail::dump_normalized_schemas(client, comm_spec, fragment_group_id,
                                  options.print_normalized_schema);
  detail::print_graph_memory_usage(client, comm_spec, fragment_group_id,
                                   options.print_memory_usage);

  if (!options.dump.empty() || (options.dump_dry_run_rounds > 0)) {
    if (options.large_vid) {
      if (options.oid_type == "string") {
        detail::dump_graph<std::string, uint64_t>(client, comm_spec,
                                                  fragment_group_id, options);
      } else if (options.oid_type == "int32") {
        detail::dump_graph<int32_t, uint64_t>(client, comm_spec,
                                              fragment_group_id, options);
      } else {
        detail::dump_graph<int64_t, uint64_t>(client, comm_spec,
                                              fragment_group_id, options);
      }
    } else {
      if (options.oid_type == "string") {
        detail::dump_graph<std::string, uint32_t>(client, comm_spec,
                                                  fragment_group_id, options);
      } else if (options.oid_type == "int32") {
        detail::dump_graph<int32_t, uint32_t>(client, comm_spec,
                                              fragment_group_id, options);
      } else {
        detail::dump_graph<int64_t, uint32_t>(client, comm_spec,
                                              fragment_group_id, options);
      }
    }
  }
}

}  // namespace vineyard

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("vineyard-graph-loader version %s\n\n",
           vineyard::vineyard_version());
    printf(R"r(Usage: loading vertices and edges as vineyard graph.

    -     ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] \
                                   <e_label_num> <efiles...> <v_label_num> <vfiles...> \
                                   [directed] [generate_eid] [retain_oid] [string_oid]

    - or: ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] --config <config.json>

          The `config.json` is a json file which should looks like

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
              "retain_oid": 1, # 0 or 1
              "string_oid": 0, # 0 or 1
              "local_vertex_map": 0 # 0 or 1
          }
)r");
    return 1;
  }

  struct vineyard::detail::loader_options options;

  int current_index = 1;
  if ((std::string(argv[current_index]) == "--socket") ||
      (std::string(argv[current_index]) == "-socket")) {
    current_index++;
    options.vineyard_ipc_socket = argv[current_index++];
    LOG(INFO) << "Using vineyard IPC socket '" << options.vineyard_ipc_socket
              << "'";
  } else {
    LOG(INFO) << "Resolve vineyard IPC socket from environment variable "
                 "'VINEYARD_IPC_SOCKET': '"
              << vineyard::read_env("VINEYARD_IPC_SOCKET") << "'";
    options.vineyard_ipc_socket = vineyard::read_env("VINEYARD_IPC_SOCKET");
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
  auto start_time = vineyard::GetMicroTimestamp();
  { vineyard::loading_vineyard_graph(options); }
  auto finish_time = vineyard::GetMicroTimestamp();
  grape::FinalizeMPIComm();

  LOG(INFO) << "Time usage: " << ((finish_time - start_time) / 1000000.0)
            << " seconds";
  LOG(INFO) << "Final peak rss = " << vineyard::get_peak_rss_pretty();

  return 0;
}
