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
#ifndef MODULES_GRAPH_TOOLS_GRAPH_LOADER_H_
#define MODULES_GRAPH_TOOLS_GRAPH_LOADER_H_

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "client/client.h"
#include "common/util/env.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/logging.h"

namespace grape {
class CommSpec;
}  // namespace grape

namespace vineyard {

namespace detail {

enum progressive_t {
  NONE,
  WHOLE,
  STEP_BY_STEP,
};

struct loader_options {
  std::string vineyard_ipc_socket;
  std::vector<std::string> efiles;
  std::vector<std::string> vfiles;
  bool directed = true;
  bool generate_eid = false;
  bool retain_oid = false;
  std::string oid_type = "int64";
  bool large_vid = true;
  bool large_eid = true;
  bool local_vertex_map = false;
  progressive_t progressive = NONE;
  bool catch_leaf_errors = true;
  std::string dump;
  size_t dump_dry_run_rounds = 0;
  bool compact_edges = false;
  bool print_memory_usage = false;
  bool print_normalized_schema = false;
  bool use_perfect_hash = false;
};

template <typename OID_T, typename VID_T>
ObjectID load_graph(Client& client, grape::CommSpec& comm_spec,
                    struct detail::loader_options const& options);

template <typename OID_T, typename VID_T>
void dump_graph(Client& client, grape::CommSpec& comm_spec,
                const ObjectID fragment_group_id,
                struct detail::loader_options const& options);

}  // namespace detail

}  // namespace vineyard

#endif  // MODULES_GRAPH_TOOLS_GRAPH_LOADER_H_
