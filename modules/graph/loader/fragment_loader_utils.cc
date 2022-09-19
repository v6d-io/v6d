/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include "graph/loader/fragment_loader_utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "basic/ds/arrow_utils.h"

#include "graph/fragment/arrow_fragment_base.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/utils/table_shuffler_beta.h"

namespace vineyard {

boost::leaf::result<std::shared_ptr<arrow::Table>> SyncSchema(
    const std::shared_ptr<arrow::Table>& table,
    const grape::CommSpec& comm_spec) {
  std::shared_ptr<arrow::Schema> local_schema =
      table != nullptr ? table->schema() : nullptr;
  std::vector<std::shared_ptr<arrow::Schema>> schemas;

  GlobalAllGatherv(local_schema, schemas, comm_spec);
  std::shared_ptr<arrow::Schema> normalized_schema;
  VY_OK_OR_RAISE(TypeLoosen(schemas, normalized_schema));

  std::shared_ptr<arrow::Table> table_out;
  if (table == nullptr) {
    VY_OK_OR_RAISE(
        vineyard::EmptyTableBuilder::Build(normalized_schema, table_out));
  } else {
    VY_OK_OR_RAISE(CastTableToSchema(table, normalized_schema, table_out));
  }
  return table_out;
}

boost::leaf::result<ObjectID> ConstructFragmentGroup(
    Client& client, ObjectID frag_id, const grape::CommSpec& comm_spec) {
  ObjectID group_object_id;
  uint64_t instance_id = client.instance_id();

  MPI_Barrier(comm_spec.comm());
  VINEYARD_DISCARD(client.SyncMetaData());

  if (comm_spec.worker_id() == 0) {
    std::vector<uint64_t> gathered_instance_ids(comm_spec.worker_num());
    std::vector<ObjectID> gathered_object_ids(comm_spec.worker_num());

    MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR,
               &gathered_instance_ids[0], sizeof(uint64_t), MPI_CHAR, 0,
               comm_spec.comm());

    MPI_Gather(&frag_id, sizeof(ObjectID), MPI_CHAR, &gathered_object_ids[0],
               sizeof(ObjectID), MPI_CHAR, 0, comm_spec.comm());

    ArrowFragmentGroupBuilder builder;
    builder.set_total_frag_num(comm_spec.fnum());
    typename ArrowFragmentBase::label_id_t vertex_label_num = 0,
                                           edge_label_num = 0;

    ObjectMeta meta;
    if (client.GetMetaData(frag_id, meta).ok()) {
      if (meta.Haskey("vertex_label_num_")) {
        vertex_label_num =
            meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
                "vertex_label_num_");
      }
      if (meta.Haskey("edge_label_num_")) {
        edge_label_num =
            meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
                "edge_label_num_");
      }
    }

    builder.set_vertex_label_num(vertex_label_num);
    builder.set_edge_label_num(edge_label_num);
    for (fid_t i = 0; i < comm_spec.fnum(); ++i) {
      builder.AddFragmentObject(
          i, gathered_object_ids[comm_spec.FragToWorker(i)],
          gathered_instance_ids[comm_spec.FragToWorker(i)]);
    }

    auto group_object =
        std::dynamic_pointer_cast<ArrowFragmentGroup>(builder.Seal(client));
    group_object_id = group_object->id();
    VY_OK_OR_RAISE(client.Persist(group_object_id));

    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  } else {
    MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR, NULL, sizeof(uint64_t),
               MPI_CHAR, 0, comm_spec.comm());
    MPI_Gather(&frag_id, sizeof(ObjectID), MPI_CHAR, NULL, sizeof(ObjectID),
               MPI_CHAR, 0, comm_spec.comm());

    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  }

  MPI_Barrier(comm_spec.comm());
  VINEYARD_DISCARD(client.SyncMetaData());
  return group_object_id;
}

}  // namespace vineyard
