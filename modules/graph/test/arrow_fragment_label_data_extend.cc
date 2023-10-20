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
#include "common/util/uuid.h"
#include "graph/fragment/graph_schema.h"
#include "graph/loader/arrow_fragment_loader.h"
#include "graph/loader/fragment_loader_utils.h"

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

    LOG(INFO) << "--------------- relabel vertex table check ...";
    for (LabelType vlabel = 0; vlabel < frag->vertex_label_num(); ++vlabel) {
      LOG(INFO) << "--------------- start dump vertex label" << vlabel
                << "---------------";
      auto iv = frag->InnerVertices(vlabel);
      for (auto v : iv) {
        LOG(INFO) << frag->GetId(v);
      }
    }

    LOG(INFO) << "--------------- relabel edge table check ...";

    for (LabelType elabel = 0; elabel < frag->edge_label_num(); ++elabel) {
      LOG(INFO) << "--------------- start dump edge label " << elabel
                << "---------------";
      for (LabelType vlabel = 0; vlabel < frag->vertex_label_num(); ++vlabel) {
        auto ie = frag->InnerVertices(vlabel);
        for (auto v : ie) {
          auto oe = frag->GetOutgoingAdjList(v, elabel);
          for (auto e : oe) {
            LOG(INFO) << frag->GetId(v) << " -> " << frag->GetId(e.neighbor())
                      << " " << static_cast<int64_t>(e.get_data<int64_t>(0))
                      << " " << static_cast<int64_t>(e.get_data<int64_t>(1))
                      << " " << static_cast<int64_t>(e.get_data<int64_t>(2))
                      << " " << static_cast<int64_t>(e.get_data<int64_t>(3));
          }
          if (frag->directed()) {
            auto ie = frag->GetIncomingAdjList(v, elabel);
            for (auto& e : ie) {
              LOG(INFO) << frag->GetId(e.neighbor()) << " -> " << frag->GetId(v)
                        << " " << static_cast<int64_t>(e.get_data<int64_t>(0))
                        << " " << static_cast<int64_t>(e.get_data<int64_t>(1))
                        << " " << static_cast<int64_t>(e.get_data<int64_t>(2))
                        << " " << static_cast<int64_t>(e.get_data<int64_t>(3));
            }
          }
        }
      }
    }
  }
}

namespace detail {

std::shared_ptr<arrow::ChunkedArray> makeInt64Array(int idx) {
  std::vector<int64_t> data(4);
  arrow::Int64Builder builder;
  int begin = idx * 4;
  for (int i = begin; i < begin + 4; i++) {
    data[i - begin] = i;
  }
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

std::vector<std::shared_ptr<arrow::Table>> makeVTables(
    int n, std::string label_name = "person") {
  std::vector<std::shared_ptr<arrow::Table>> tables;
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
      schema, {makeInt64Array(n), makeInt64Array(n), makeInt64Array(n),
               makeInt64Array(n), makeInt64Array(n)});
  tables.push_back(table);
  return {tables};
}

std::vector<std::vector<std::shared_ptr<arrow::Table>>> makeETables(
    int n, std::string label_name = "knows") {
  std::vector<std::shared_ptr<arrow::Table>> tables;
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
      schema, {makeInt64Array(n), makeInt64Array(n), makeInt64Array(n),
               makeInt64Array(n), makeInt64Array(n), makeInt64Array(n)});
  tables.push_back(table);
  return {tables};
}
}  // namespace detail

int main(int argc, char** argv) {
  if (argc < 2) {
    printf(
        "usage: ./arrow_fragment_label_data_extend <ipc_socket> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  using loader_t = ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                       property_graph_types::VID_TYPE>;

  grape::InitMPIComm();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);
    vineyard::ObjectID frag_group_id = vineyard::InvalidObjectID();

    // first construct a basic graph
    {
      auto vtables = ::detail::makeVTables(0);
      auto etables = ::detail::makeETables(0);
      auto loader = std::make_unique<loader_t>(
          client, comm_spec, vtables, etables, directed != 0,
          /*generate_eid*/ false, /*retain_oid*/ true);
      frag_group_id = loader->LoadFragmentAsFragmentGroup().value();
      WriteOut(client, comm_spec, frag_group_id);
    }

    for (int i = 1; i < 3; ++i) {
      auto vtables = ::detail::makeVTables(i);
      auto etables = std::vector<std::vector<std::shared_ptr<arrow::Table>>>();
      auto loader = std::make_unique<loader_t>(
          client, comm_spec, vtables, etables, directed != 0,
          /*generate_eid*/ false, /*retain_oid*/ true);
      std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
          std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
              client.GetObject(frag_group_id));
      auto locations = fg->FragmentLocations();
      for (const auto& pair : fg->Fragments()) {
        if (locations.at(pair.first) != client.instance_id()) {
          continue;
        }
        auto frag_id = pair.second;
        frag_id = loader->AddDataToExistedVLabel(frag_id, 0).value();
        frag_group_id =
            ConstructFragmentGroup(client, frag_id, comm_spec).value();
      }
    }
    LOG(INFO) << "[START DUMP FRAGMENT GROUP AFTER INCREMENTAL ADD VERTICES]";
    WriteOut(client, comm_spec, frag_group_id);
    LOG(INFO) << "[END FRAGMENT GROUP AFTER INCREMENTAL ADD VERTICES]";

    for (int i = 1; i < 3; ++i) {
      auto vtables = std::vector<std::shared_ptr<arrow::Table>>();
      auto etables = ::detail::makeETables(i);
      auto loader = std::make_unique<loader_t>(
          client, comm_spec, vtables, etables, directed != 0,
          /*generate_eid*/ false, /*retain_oid*/ true);
      std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
          std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
              client.GetObject(frag_group_id));
      auto locations = fg->FragmentLocations();
      for (const auto& pair : fg->Fragments()) {
        if (locations.at(pair.first) != client.instance_id()) {
          continue;
        }
        auto frag_id = pair.second;
        frag_id = loader->AddDataToExistedELabel(frag_id, 0).value();
        frag_group_id =
            ConstructFragmentGroup(client, frag_id, comm_spec).value();
      }
      WriteOut(client, comm_spec, frag_group_id);
    }
    LOG(INFO) << "[START DUMP FRAGMENT GROUP AFTER INCREMENTAL ADD EDGES]";
    WriteOut(client, comm_spec, frag_group_id);
    LOG(INFO) << "[END DUMP FRAGMENT GROUP AFTER INCREMENTAL ADD EDGES]";
  }
  grape::FinalizeMPIComm();

  LOG(INFO) << "Passed arrow fragment relabel test...";

  return 0;
}
