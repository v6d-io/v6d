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

struct loader_options {
  std::string vineyard_ipc_socket;
  std::vector<std::string> efiles;
  std::vector<std::string> vfiles;
  bool directed = true;
  bool generate_eid = false;
  bool string_oid = false;
  bool int32_oid = false;
  bool local_vertex_map = false;
  bool catch_leaf_errors = true;
};

ObjectID loading_vineyard_graph_int32(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

ObjectID loading_vineyard_graph_int64(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

ObjectID loading_vineyard_graph_string(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

ObjectID loading_vineyard_graph_int32_localvm(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

ObjectID loading_vineyard_graph_int64_localvm(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

ObjectID loading_vineyard_graph_string_localvm(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

}  // namespace detail

}  // namespace vineyard

#endif  // MODULES_GRAPH_TOOLS_GRAPH_LOADER_H_
