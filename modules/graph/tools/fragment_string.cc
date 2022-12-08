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
#include "graph/tools/graph_loader.h"

namespace vineyard {

namespace detail {

ObjectID loading_vineyard_graph_string(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options) {
  MPI_Barrier(comm_spec.comm());
  auto loader = std::make_unique<ArrowFragmentLoader<
      std::string, property_graph_types::VID_TYPE, ArrowVertexMap>>(
      client, comm_spec, options.efiles, options.vfiles, options.directed != 0,
      options.generate_eid != 0);

  if (options.catch_leaf_errors) {
    return boost::leaf::try_handle_all(
        [&loader]() { return loader->LoadFragmentAsFragmentGroup(); },
        [](const GSError& e) {
          LOG(FATAL) << e.error_msg;
          return InvalidObjectID();
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(FATAL) << "Unmatched error " << unmatched;
          return InvalidObjectID();
        });
  } else {
    return loader->LoadFragmentAsFragmentGroup().value();
  }
}

ObjectID loading_vineyard_graph_string_localvm(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options) {
  MPI_Barrier(comm_spec.comm());
  auto loader = std::make_unique<ArrowFragmentLoader<
      std::string, property_graph_types::VID_TYPE, ArrowLocalVertexMap>>(
      client, comm_spec, options.efiles, options.vfiles, options.directed != 0,
      options.generate_eid != 0);

  if (options.catch_leaf_errors) {
    return boost::leaf::try_handle_all(
        [&loader]() { return loader->LoadFragmentAsFragmentGroup(); },
        [](const GSError& e) {
          LOG(FATAL) << e.error_msg;
          return InvalidObjectID();
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(FATAL) << "Unmatched error " << unmatched;
          return InvalidObjectID();
        });
  } else {
    return loader->LoadFragmentAsFragmentGroup().value();
  }
}

}  // namespace detail

}  // namespace vineyard
