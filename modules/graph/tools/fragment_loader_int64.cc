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
#include "graph/tools/fragment_loader_impl.h"
#include "graph/tools/graph_loader.h"

namespace vineyard {

namespace detail {

template ObjectID load_graph<int64_t, uint32_t>(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

template void dump_graph<int64_t, uint32_t>(
    Client& client, grape::CommSpec& comm_spec,
    const ObjectID fragment_group_id,
    struct detail::loader_options const& options);

template ObjectID load_graph<int64_t, uint64_t>(
    Client& client, grape::CommSpec& comm_spec,
    struct detail::loader_options const& options);

template void dump_graph<int64_t, uint64_t>(
    Client& client, grape::CommSpec& comm_spec,
    const ObjectID fragment_group_id,
    struct detail::loader_options const& options);

}  // namespace detail

}  // namespace vineyard
