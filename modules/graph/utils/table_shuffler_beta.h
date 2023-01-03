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

#ifndef MODULES_GRAPH_UTILS_TABLE_SHUFFLER_BETA_H_
#define MODULES_GRAPH_UTILS_TABLE_SHUFFLER_BETA_H_

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "grape/communication/sync_comm.h"
#include "grape/utils/concurrent_queue.h"
#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/table_shuffler.h"

namespace vineyard {

namespace beta {

// for source-level backwards compatibility

using vineyard::RecvArrowBuffer;
using vineyard::SendArrowBuffer;

using vineyard::CheckSchemaConsistency;

using vineyard::SerializeSelectedItems;
using vineyard::SerializeSelectedRows;

using vineyard::DeserializeSelectedItems;
using vineyard::DeserializeSelectedRows;

using vineyard::SelectItems;
using vineyard::SelectRows;

using vineyard::ShufflePropertyEdgeTable;
using vineyard::ShufflePropertyEdgeTableByPartition;
using vineyard::ShufflePropertyVertexTable;
using vineyard::ShuffleTableByOffsetLists;

}  // namespace beta

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_SHUFFLER_BETA_H_
