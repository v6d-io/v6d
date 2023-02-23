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

#include <fstream>
#include <string>

#include "client/client.h"
#include "common/util/logging.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/graph_schema.h"
#include "graph/loader/arrow_fragment_loader.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;

void WriteOut(vineyard::Client& client, const grape::CommSpec& comm_spec,
              vineyard::ObjectID fragment_group_id) {
  LOG(INFO) << "Loaded graph to vineyard: " << fragment_group_id;
  std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(fragment_group_id));

  for (const auto& pair : fg->Fragments()) {
    LOG(INFO) << "[frag-" << pair.first << "]: " << pair.second;
  }

  // NB: only retrieve local fragments.
  auto locations = fg->FragmentLocations();
  for (const auto& pair : fg->Fragments()) {
    if (locations.at(pair.first) != client.instance_id()) {
      continue;
    }
    auto frag_id = pair.second;
    auto frag = std::dynamic_pointer_cast<GraphType>(client.GetObject(frag_id));
    auto schema = frag->schema();
    auto mg_schema = vineyard::MaxGraphSchema(schema);
    mg_schema.DumpToFile("/tmp/" + std::to_string(fragment_group_id) + ".json");

    LOG(INFO) << "graph total node number: " << frag->GetTotalNodesNum();
    LOG(INFO) << "fragment edge number: " << frag->GetEdgeNum();
    LOG(INFO) << "fragment in edge number: " << frag->GetInEdgeNum();
    LOG(INFO) << "fragment out edge number: " << frag->GetOutEdgeNum();

    LOG(INFO) << "[worker-" << comm_spec.worker_id()
              << "] loaded graph to vineyard: " << ObjectIDToString(frag_id)
              << " ...";

    for (LabelType vlabel = 0; vlabel < frag->vertex_label_num(); ++vlabel) {
      LOG(INFO) << "vertex table: " << vlabel << " -> "
                << frag->vertex_data_table(vlabel)->schema()->ToString();
    }
    for (LabelType elabel = 0; elabel < frag->edge_label_num(); ++elabel) {
      LOG(INFO) << "edge table: " << elabel << " -> "
                << frag->edge_data_table(elabel)->schema()->ToString();
    }

    LOG(INFO) << "--------------- consolidate vertex/edge table columns ...";

    if (frag->vertex_data_table(0)->columns().size() >= 4) {
      auto vcols = std::vector<std::string>{"value1", "value3"};
      auto vfrag_id =
          frag->ConsolidateVertexColumns(client, 0, vcols, "vmerged").value();
      auto vfrag =
          std::dynamic_pointer_cast<GraphType>(client.GetObject(vfrag_id));

      for (LabelType vlabel = 0; vlabel < vfrag->vertex_label_num(); ++vlabel) {
        LOG(INFO) << "vertex table: " << vlabel << " -> "
                  << vfrag->vertex_data_table(vlabel)->schema()->ToString();
      }
    }

    if (frag->edge_data_table(0)->columns().size() >= 4) {
      auto ecols = std::vector<std::string>{"value2", "value4"};
      auto efrag_id =
          frag->ConsolidateEdgeColumns(client, 0, ecols, "emerged").value();
      auto efrag =
          std::dynamic_pointer_cast<GraphType>(client.GetObject(efrag_id));

      for (LabelType elabel = 0; elabel < efrag->edge_label_num(); ++elabel) {
        LOG(INFO) << "edge table: " << elabel << " -> "
                  << efrag->edge_data_table(elabel)->schema()->ToString();
      }
    }
  }
}

namespace detail {

std::shared_ptr<arrow::ChunkedArray> makeInt64Array() {
  std::vector<int64_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  arrow::Int64Builder builder;
  CHECK_ARROW_ERROR(builder.AppendValues(data));
  std::shared_ptr<arrow::Array> out;
  CHECK_ARROW_ERROR(builder.Finish(&out));
  return arrow::ChunkedArray::Make({out}).ValueOrDie();
}

std::shared_ptr<arrow::Schema> attachMetadata(
    std::shared_ptr<arrow::Schema> schema, std::string const& key,
    std::string const& value) {
  std::shared_ptr<arrow::KeyValueMetadata> metadata;
  if (schema->HasMetadata()) {
    metadata = schema->metadata()->Copy();
  } else {
    metadata = std::make_shared<arrow::KeyValueMetadata>();
  }
  metadata->Append(key, value);
  return schema->WithMetadata(metadata);
}

std::vector<std::shared_ptr<arrow::Table>> makeVTables() {
  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          arrow::field("id", arrow::int64()),
          arrow::field("value1", arrow::int64()),
          arrow::field("value2", arrow::int64()),
          arrow::field("value3", arrow::int64()),
          arrow::field("value4", arrow::int64()),
      });
  schema = attachMetadata(schema, "label", "person");
  auto table = arrow::Table::Make(
      schema, {makeInt64Array(), makeInt64Array(), makeInt64Array(),
               makeInt64Array(), makeInt64Array()});
  return {table};
}

std::vector<std::vector<std::shared_ptr<arrow::Table>>> makeETables() {
  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          arrow::field("src_id", arrow::int64()),
          arrow::field("dst_id", arrow::int64()),
          arrow::field("value1", arrow::int64()),
          arrow::field("value2", arrow::int64()),
          arrow::field("value3", arrow::int64()),
          arrow::field("value4", arrow::int64()),
      });
  schema = attachMetadata(schema, "label", "knows");
  schema = attachMetadata(schema, "src_label", "person");
  schema = attachMetadata(schema, "dst_label", "person");
  auto table = arrow::Table::Make(
      schema, {makeInt64Array(), makeInt64Array(), makeInt64Array(),
               makeInt64Array(), makeInt64Array(), makeInt64Array()});
  return {{table}};
}
}  // namespace detail

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: ./arrow_fragment_test <ipc_socket> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  auto vtables = ::detail::makeVTables();
  auto etables = ::detail::makeETables();

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    {
      auto loader =
          std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                               property_graph_types::VID_TYPE>>(
              client, comm_spec, vtables, etables, directed != 0);
      vineyard::ObjectID fragment_group_id =
          loader->LoadFragmentAsFragmentGroup().value();
      WriteOut(client, comm_spec, fragment_group_id);
    }

    // Load from efiles
    {
      auto loader =
          std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                               property_graph_types::VID_TYPE>>(
              client, comm_spec, etables, directed != 0);
      vineyard::ObjectID fragment_group_id =
          loader->LoadFragmentAsFragmentGroup().value();
      WriteOut(client, comm_spec, fragment_group_id);
    }
  }
  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment test...";

  return 0;
}
