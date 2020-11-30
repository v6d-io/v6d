/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/util/config.h"
#include "arrow/util/key_value_metadata.h"

#include "grape/worker/comm_spec.h"

#include "basic/stream/dataframe_stream.h"
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "io/io/io_factory.h"
#include "io/io/local_io_adaptor.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/loader/basic_arrow_fragment_loader.h"
#include "graph/utils/error.h"
#include "graph/utils/partitioner.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_vertex_map.h"

#define HASH_PARTITION

namespace vineyard {

template <typename OID_T = property_graph_types::OID_TYPE,
          typename VID_T = property_graph_types::VID_TYPE>
class ArrowFragmentLoader {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using vertex_map_t = ArrowVertexMap<internal_oid_t, vid_t>;
  // These consts represent which column in the arrow table represents oid.
  const int id_column = 0;
  const int src_column = 0;
  const int dst_column = 1;
  const int edge_id_column = 2;
  // These consts represent the key in the path of vfile/efile
  static constexpr const char* LABEL_TAG = "label";
  static constexpr const char* SRC_LABEL_TAG = "src_label";
  static constexpr const char* DST_LABEL_TAG = "dst_label";
#ifdef HASH_PARTITION
  using partitioner_t = HashPartitioner<oid_t>;
#else
  using partitioner_t = SegmentedPartitioner<oid_t>;
#endif
  using basic_loader_t = BasicArrowFragmentLoader<oid_t, vid_t, partitioner_t>;

 public:
  /**
   *
   * @param client
   * @param comm_spec
   * @param efiles An example of efile:
   * /data/twitter_e_0_0_0#src_label=v0&dst_label=v0&label=e0;/data/twitter_e_0_1_0#src_label=v0&dst_label=v1&label=e0;/data/twitter_e_1_0_0#src_label=v1&dst_label=v0&label=e0;/data/twitter_e_1_1_0#src_label=v1&dst_label=v1&label=e0
   * @param vfiles An example of vfile: /data/twitter_v_0#label=v0
   * @param directed
   */
  ArrowFragmentLoader(vineyard::Client& client,
                      const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      const std::vector<std::string>& vfiles,
                      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        efiles_(efiles),
        vfiles_(vfiles),
        vertex_label_num_(vfiles.size()),
        edge_label_num_(efiles.size()),
        directed_(directed),
        generate_eid_(generate_eid),
        basic_arrow_fragment_loader_(comm_spec) {}

  ArrowFragmentLoader(vineyard::Client& client,
                      const grape::CommSpec& comm_spec,
                      const std::vector<ObjectID>& vstreams,
                      const std::vector<std::vector<ObjectID>>& estreams,
                      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        v_streams_(vstreams),
        e_streams_(estreams),
        directed_(directed),
        generate_eid_(generate_eid),
        basic_arrow_fragment_loader_(comm_spec) {}

  ArrowFragmentLoader(
      vineyard::Client& client, const grape::CommSpec& comm_spec,
      label_id_t vertex_label_num, label_id_t edge_label_num,
      std::vector<std::shared_ptr<arrow::Table>> const& partial_v_tables,
      std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
          partial_e_tables,
      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        vertex_label_num_(vertex_label_num),
        edge_label_num_(edge_label_num),
        partial_v_tables_(partial_v_tables),
        partial_e_tables_(partial_e_tables),
        directed_(directed),
        generate_eid_(generate_eid),
        basic_arrow_fragment_loader_(comm_spec) {}

  ArrowFragmentLoader(vineyard::Client& client,
                      const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        efiles_(efiles),
        vfiles_(),
        vertex_label_num_(0),
        edge_label_num_(efiles.size()),
        directed_(directed),
        generate_eid_(generate_eid),
        basic_arrow_fragment_loader_(comm_spec) {}

  ~ArrowFragmentLoader() = default;

  boost::leaf::result<vineyard::ObjectID> LoadFragment() {
#if defined(WITH_PROFILING)
    double start_ts = GetCurrentTime();
#endif
    BOOST_LEAF_CHECK(initPartitioner());
#if defined(WITH_PROFILING)
    double init_partitioner_ts = GetCurrentTime();
    VLOG(1) << "initPartitioner uses " << (init_partitioner_ts - start_ts)
            << " seconds";
#endif
    BOOST_LEAF_CHECK(initBasicLoader());
#if defined(WITH_PROFILING)
    double init_basic_loader_ts = GetCurrentTime();
    VLOG(1) << "initBasicLoader uses "
            << (init_basic_loader_ts - init_partitioner_ts) << " seconds";
#endif
    BOOST_LEAF_AUTO(frag_id, shuffleAndBuild());
#if defined(WITH_PROFILING)
    double shuffle_and_build_ts = GetCurrentTime();
    VLOG(1) << "shuffleAndBuild uses "
            << (shuffle_and_build_ts - init_basic_loader_ts) << " seconds";
    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "] load fragments use: " << (shuffle_and_build_ts - start_ts)
            << " seconds";
#endif
    return frag_id;
  }

  boost::leaf::result<vineyard::ObjectID> LoadFragmentAsFragmentGroup() {
    BOOST_LEAF_AUTO(frag_id, LoadFragment());
    BOOST_LEAF_AUTO(group_id,
                    constructFragmentGroup(client_, frag_id, comm_spec_,
                                           vertex_label_num_, edge_label_num_));
    return group_id;
  }

 protected:  // for subclasses
  boost::leaf::result<void> initPartitioner() {
#ifdef HASH_PARTITION
    partitioner_.Init(comm_spec_.fnum());
#else
    if (vfiles_.empty()) {
      RETURN_GS_ERROR(
          ErrorCode::kInvalidOperationError,
          "Segmented partitioner is not supported when the v-file is "
          "not provided");
    }
    std::vector<std::shared_ptr<arrow::Table>> vtables;
    {
      BOOST_LEAF_AUTO(tmp, loadVertexTables(vfiles_, 0, 1));
      vtables = tmp;
    }
    std::vector<oid_t> oid_list;

    for (auto& table : vtables) {
      std::shared_ptr<arrow::ChunkedArray> oid_array_chunks =
          table->column(id_column);
      size_t chunk_num = oid_array_chunks->num_chunks();

      for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
        std::shared_ptr<oid_array_t> array =
            std::dynamic_pointer_cast<oid_array_t>(
                oid_array_chunks->chunk(chunk_i));
        int64_t length = array->length();
        for (int64_t i = 0; i < length; ++i) {
          oid_list.emplace_back(oid_t(array->GetView(i)));
        }
      }
    }

    partitioner_.Init(comm_spec_.fnum(), oid_list);
#endif
    return {};
  }

  boost::leaf::result<void> initBasicLoader() {
    std::vector<std::shared_ptr<arrow::Table>> partial_v_tables;
    std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables;
    if (!v_streams_.empty() && !e_streams_.empty()) {
      BOOST_LEAF_ASSIGN(partial_v_tables, gatherVTables(client_, v_streams_));
      BOOST_LEAF_ASSIGN(partial_e_tables, gatherETables(client_, e_streams_));
      // note that batches of multiple labels may comes in the same stream.
      vertex_label_num_ = partial_v_tables.size();
      edge_label_num_ = partial_e_tables.size();
    } else if (!partial_v_tables_.empty() && !partial_e_tables_.empty()) {
      for (size_t vlabel = 0; vlabel < partial_v_tables_.size(); ++vlabel) {
        std::shared_ptr<arrow::Table> result_table;
        VY_OK_OR_RAISE(rebuildVTableMetadata(vlabel, partial_v_tables_[vlabel],
                                             result_table));
        partial_v_tables.emplace_back(result_table);
      }
      for (size_t elabel = 0; elabel < partial_e_tables_.size(); ++elabel) {
        std::vector<std::shared_ptr<arrow::Table>> subetables;
        for (auto const& etable : partial_e_tables_[elabel]) {
          std::shared_ptr<arrow::Table> result_table;
          VY_OK_OR_RAISE(rebuildETableMetadata(
              elabel, partial_e_tables_[elabel].size(), etable, result_table));
          subetables.emplace_back(result_table);
        }
        partial_e_tables.emplace_back(subetables);
      }
    } else {
      // if vfiles is empty, we infer oids from efile
      if (vfiles_.empty()) {
        possible_duplicate_oid = true;
        auto load_procedure = [&]() {
          return loadEVTablesFromEFiles(efiles_, comm_spec_.worker_id(),
                                        comm_spec_.worker_num());
        };
        BOOST_LEAF_AUTO(ev_tables, sync_gs_error(comm_spec_, load_procedure));
        partial_v_tables = ev_tables.first;
        partial_e_tables = ev_tables.second;
      } else {
        auto load_v_procedure = [&]() {
          return loadVertexTables(vfiles_, comm_spec_.worker_id(),
                                  comm_spec_.worker_num());
        };
        BOOST_LEAF_ASSIGN(partial_v_tables,
                          sync_gs_error(comm_spec_, load_v_procedure));
        auto load_e_procedure = [&]() {
          return loadEdgeTables(efiles_, comm_spec_.worker_id(),
                                comm_spec_.worker_num());
        };
        BOOST_LEAF_ASSIGN(partial_e_tables,
                          sync_gs_error(comm_spec_, load_e_procedure));
      }
    }

    if (generate_eid_) {
      generateEdgeId(partial_e_tables);
    }

    basic_arrow_fragment_loader_.Init(partial_v_tables, partial_e_tables);
    basic_arrow_fragment_loader_.SetPartitioner(partitioner_);

    return {};
  }

  boost::leaf::result<void> generateEdgeId(
      std::vector<std::vector<std::shared_ptr<arrow::Table>>>& edge_tables) {
    IdParser<uint64_t> eid_parser;
    eid_parser.Init(comm_spec_.fnum(), edge_label_num_);
    for (label_id_t e_label = 0; e_label < edge_label_num_; ++e_label) {
      auto& edge_table_list = edge_tables[e_label];
      uint64_t cur_id = eid_parser.GenerateId(comm_spec_.fid(), e_label, 0);
      for (size_t edge_table_index = 0;
           edge_table_index != edge_table_list.size(); ++edge_table_index) {
        auto& edge_table = edge_table_list[edge_table_index];
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        VY_OK_OR_RAISE(TableToRecordBatches(edge_table, &batches));
        std::vector<std::shared_ptr<arrow::Array>> generated_arrays;
        for (auto& rb : batches) {
          int64_t row_num = rb->num_rows();
          typename ConvertToArrowType<int64_t>::BuilderType builder;
          for (int64_t i = 0; i != row_num; ++i) {
            builder.Append(static_cast<int64_t>(cur_id));
            ++cur_id;
          }
          std::shared_ptr<arrow::Array> eid_array;
          builder.Finish(&eid_array);
          generated_arrays.push_back(eid_array);
        }
        std::shared_ptr<arrow::ChunkedArray> chunked_eid_array =
            std::make_shared<arrow::ChunkedArray>(
                generated_arrays, ConvertToArrowType<int64_t>::TypeValue());

        auto eid_field = std::make_shared<arrow::Field>(
            "eid", ConvertToArrowType<int64_t>::TypeValue());

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
        CHECK_ARROW_ERROR(edge_table->AddColumn(
            edge_id_column, eid_field, chunked_eid_array, &edge_table));
#else
        CHECK_ARROW_ERROR_AND_ASSIGN(
            edge_table, edge_table->AddColumn(edge_id_column, eid_field,
                                              chunked_eid_array));
#endif
      }
    }

    return {};
  }

  boost::leaf::result<vineyard::ObjectID> shuffleAndBuild() {
#if defined(WITH_PROFILING)
    auto start_ts = GetCurrentTime();
#endif
    // When vfiles_ is empty, it means we build vertex table from efile
    BOOST_LEAF_AUTO(local_v_tables,
                    basic_arrow_fragment_loader_.ShuffleVertexTables(
                        possible_duplicate_oid));
#if defined(WITH_PROFILING)
    auto shuffle_vtable_ts = GetCurrentTime();
    VLOG(1) << "ShuffleVertexTables uses " << (shuffle_vtable_ts - start_ts)
            << " seconds";
#endif
    auto oid_lists = basic_arrow_fragment_loader_.GetOidLists();
    BasicArrowVertexMapBuilder<typename InternalType<oid_t>::type, vid_t>
        vm_builder(client_, comm_spec_.fnum(), vertex_label_num_, oid_lists);
    auto vm = vm_builder.Seal(client_);
    auto vm_ptr =
        std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(vm->id()));
    auto mapper = [&vm_ptr](fid_t fid, label_id_t label, internal_oid_t oid,
                            vid_t& gid) {
      CHECK(vm_ptr->GetGid(fid, label, oid, gid));
      return true;
    };

#if defined(WITH_PROFILING)
    auto build_vertex_map_ts = GetCurrentTime();
    VLOG(1) << "Build vertex map uses "
            << (build_vertex_map_ts - shuffle_vtable_ts) << " seconds";
#endif
    BOOST_LEAF_AUTO(local_e_tables,
                    basic_arrow_fragment_loader_.ShuffleEdgeTables(mapper));
#if defined(WITH_PROFILING)
    auto shuffle_etable_ts = GetCurrentTime();
    VLOG(1) << "ShuffleEdgeTables uses "
            << (shuffle_etable_ts - build_vertex_map_ts) << " seconds";
#endif
    BasicArrowFragmentBuilder<oid_t, vid_t> frag_builder(client_, vm_ptr);
    PropertyGraphSchema schema;

    schema.set_fnum(comm_spec_.fnum());

    {
      std::vector<std::string> vertex_label_list(vertex_label_num_);
      std::vector<bool> vertex_label_bitset(vertex_label_num_, false);
      for (auto& pair : vertex_label_to_index_) {
        if (pair.second > vertex_label_num_) {
          RETURN_GS_ERROR(ErrorCode::kIOError,
                          "Failed to map vertex label to index");
        }
        if (vertex_label_bitset[pair.second]) {
          RETURN_GS_ERROR(ErrorCode::kIOError,
                          "Multiple vertex labels are mapped to one index.");
        }
        vertex_label_bitset[pair.second] = true;
        vertex_label_list[pair.second] = pair.first;
      }
      for (label_id_t v_label = 0; v_label != vertex_label_num_; ++v_label) {
        std::string vertex_label = vertex_label_list[v_label];
        auto entry = schema.CreateEntry(vertex_label, "VERTEX");

        std::unordered_map<std::string, std::string> kvs;
        auto table = local_v_tables[v_label];
        table->schema()->metadata()->ToUnorderedMap(&kvs);

        entry->AddPrimaryKeys(1, std::vector<std::string>{kvs["primary_key"]});

        // N.B. ID column is not removed, and we need that
        for (int64_t i = 0; i < table->num_columns(); ++i) {
          entry->AddProperty(table->schema()->field(i)->name(),
                             table->schema()->field(i)->type());
        }
      }
    }

    {
      std::vector<std::string> edge_label_list(edge_label_num_);
      std::vector<bool> edge_label_bitset(edge_label_num_, false);
      for (auto& pair : edge_label_to_index_) {
        if (pair.second > edge_label_num_) {
          RETURN_GS_ERROR(ErrorCode::kIOError,
                          "Failed to map edge label to index");
        }
        if (edge_label_bitset[pair.second]) {
          RETURN_GS_ERROR(ErrorCode::kIOError,
                          "Multiple edge labels are mapped to one index.");
        }
        edge_label_bitset[pair.second] = true;
        edge_label_list[pair.second] = pair.first;
      }
      for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
        std::string edge_label = edge_label_list[e_label];
        auto entry = schema.CreateEntry(edge_label, "EDGE");
        auto& pairs = edge_vertex_label_.at(edge_label);
        for (auto& vpair : pairs) {
          std::string src_label = vpair.first;
          std::string dst_label = vpair.second;
          entry->AddRelation(src_label, dst_label);
        }

        auto table = local_e_tables.at(e_label);
        for (int64_t i = 2; i < table->num_columns(); ++i) {
          entry->AddProperty(table->schema()->field(i)->name(),
                             table->schema()->field(i)->type());
        }
      }
    }

    frag_builder.SetPropertyGraphSchema(std::move(schema));

    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();
#if defined(WITH_PROFILING)
    auto frag_builder_start_ts = GetCurrentTime();
#endif
    BOOST_LEAF_CHECK(frag_builder.Init(
        comm_spec_.fid(), comm_spec_.fnum(), std::move(local_v_tables),
        std::move(local_e_tables), directed_, thread_num));
#if defined(WITH_PROFILING)
    auto frag_builder_init_ts = GetCurrentTime();
    VLOG(1) << "Init frag builder uses "
            << (frag_builder_init_ts - frag_builder_start_ts) << " seconds";
#endif
    auto frag = std::dynamic_pointer_cast<ArrowFragment<oid_t, vid_t>>(
        frag_builder.Seal(client_));
#if defined(WITH_PROFILING)
    auto frag_builder_seal_ts = GetCurrentTime();
    VLOG(1) << "Seal frag builder uses "
            << (frag_builder_seal_ts - frag_builder_init_ts) << " seconds";
#endif
    VINEYARD_CHECK_OK(client_.Persist(frag->id()));
#if defined(WITH_PROFILING)
    auto frag_builder_persist_ts = GetCurrentTime();
    VLOG(1) << "Persist frag builder uses "
            << (frag_builder_persist_ts - frag_builder_seal_ts) << " seconds";
#endif
    return frag->id();
  }

  boost::leaf::result<vineyard::ObjectID> constructFragmentGroup(
      vineyard::Client& client, vineyard::ObjectID frag_id,
      const grape::CommSpec& comm_spec, label_id_t v_label_num,
      label_id_t e_label_num) {
    vineyard::ObjectID group_object_id;
    uint64_t instance_id = client.instance_id();

    if (comm_spec.worker_id() == 0) {
      std::vector<uint64_t> gathered_instance_ids(comm_spec.worker_num());
      std::vector<vineyard::ObjectID> gathered_object_ids(
          comm_spec.worker_num());

      MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR,
                 &gathered_instance_ids[0], sizeof(uint64_t), MPI_CHAR, 0,
                 comm_spec.comm());

      MPI_Gather(&frag_id, sizeof(vineyard::ObjectID), MPI_CHAR,
                 &gathered_object_ids[0], sizeof(vineyard::ObjectID), MPI_CHAR,
                 0, comm_spec.comm());

      ArrowFragmentGroupBuilder builder;
      builder.set_total_frag_num(comm_spec.fnum());
      builder.set_vertex_label_num(v_label_num);
      builder.set_edge_label_num(e_label_num);
      for (fid_t i = 0; i < comm_spec.fnum(); ++i) {
        builder.AddFragmentObject(
            i, gathered_object_ids[comm_spec.FragToWorker(i)],
            gathered_instance_ids[comm_spec.FragToWorker(i)]);
      }

      auto group_object =
          std::dynamic_pointer_cast<ArrowFragmentGroup>(builder.Seal(client));
      group_object_id = group_object->id();
      VY_OK_OR_RAISE(client.Persist(group_object_id));

      MPI_Bcast(&group_object_id, sizeof(vineyard::ObjectID), MPI_CHAR, 0,
                comm_spec.comm());

    } else {
      MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR, NULL,
                 sizeof(uint64_t), MPI_CHAR, 0, comm_spec.comm());
      MPI_Gather(&frag_id, sizeof(vineyard::ObjectID), MPI_CHAR, NULL,
                 sizeof(vineyard::ObjectID), MPI_CHAR, 0, comm_spec.comm());

      MPI_Bcast(&group_object_id, sizeof(vineyard::ObjectID), MPI_CHAR, 0,
                comm_spec.comm());
    }
    return group_object_id;
  }

  boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
  loadVertexTables(const std::vector<std::string>& files, int index,
                   int total_parts) {
    auto label_num = static_cast<label_id_t>(files.size());
    std::vector<std::shared_ptr<arrow::Table>> tables(label_num);

    for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
      std::unique_ptr<vineyard::LocalIOAdaptor,
                      std::function<void(vineyard::LocalIOAdaptor*)>>
          io_adaptor(new vineyard::LocalIOAdaptor(files[label_id] +
                                                  "#header_row=true"),
                     io_deleter_);
      auto read_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
        VY_OK_OR_RAISE(io_adaptor->Open());
        std::shared_ptr<arrow::Table> table;
        VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
        return table;
      };

      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

      auto sync_schema_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        return SyncSchema(table, comm_spec_);
      };

      BOOST_LEAF_AUTO(normalized_table,
                      sync_gs_error(comm_spec_, sync_schema_procedure));

      auto meta = std::make_shared<arrow::KeyValueMetadata>();

      meta->Append("type", "VERTEX");
      meta->Append(basic_loader_t::ID_COLUMN, std::to_string(id_column));

      auto adaptor_meta = io_adaptor->GetMeta();
      // Check if label name is in meta
      if (adaptor_meta.find(LABEL_TAG) == adaptor_meta.end()) {
        RETURN_GS_ERROR(
            ErrorCode::kIOError,
            "Metadata of input vertex files should contain label name");
      }
      auto v_label_name = adaptor_meta.find(LABEL_TAG)->second;

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
      std::unordered_map<std::string, std::string> metakv;
      meta->ToUnorderedMap(&metakv);
      for (auto const& kv : adaptor_meta) {
        metakv[kv.first] = kv.second;
      }
      meta = std::make_shared<arrow::KeyValueMetadata>();
      for (auto const& kv : metakv) {
        meta->Append(kv.first, kv.second);
      }
#else
      for (auto const& kv : adaptor_meta) {
        CHECK_ARROW_ERROR(meta->Set(kv.first, kv.second));
      }
#endif

      tables[label_id] = normalized_table->ReplaceSchemaMetadata(meta);
      vertex_label_to_index_[v_label_name] = label_id;
    }
    return tables;
  }

  boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
  loadEdgeTables(const std::vector<std::string>& files, int index,
                 int total_parts) {
    auto label_num = static_cast<label_id_t>(files.size());
    std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables(label_num);

    try {
      for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
        std::vector<std::string> sub_label_files;
        boost::split(sub_label_files, files[label_id], boost::is_any_of(";"));

        for (size_t j = 0; j < sub_label_files.size(); ++j) {
          std::unique_ptr<vineyard::LocalIOAdaptor,
                          std::function<void(vineyard::LocalIOAdaptor*)>>
              io_adaptor(new vineyard::LocalIOAdaptor(sub_label_files[j] +
                                                      "#header_row=true"),
                         io_deleter_);
          auto read_procedure =
              [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
            VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
            VY_OK_OR_RAISE(io_adaptor->Open());
            std::shared_ptr<arrow::Table> table;
            VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
            return table;
          };
          BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

          auto sync_schema_procedure =
              [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
            return SyncSchema(table, comm_spec_);
          };
          BOOST_LEAF_AUTO(normalized_table,
                          sync_gs_error(comm_spec_, sync_schema_procedure));

          std::shared_ptr<arrow::KeyValueMetadata> meta(
              new arrow::KeyValueMetadata());
          meta->Append("type", "EDGE");
          meta->Append(basic_loader_t::SRC_COLUMN, std::to_string(src_column));
          meta->Append(basic_loader_t::DST_COLUMN, std::to_string(dst_column));
          meta->Append("sub_label_num", std::to_string(sub_label_files.size()));

          auto adaptor_meta = io_adaptor->GetMeta();
          auto it = adaptor_meta.find(LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain label name");
          }
          std::string edge_label_name = it->second;

          it = adaptor_meta.find(SRC_LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain src label name");
          }
          std::string src_label_name = it->second;

          it = adaptor_meta.find(DST_LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain dst label name");
          }
          std::string dst_label_name = it->second;

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
          std::unordered_map<std::string, std::string> metakv;
          meta->ToUnorderedMap(&metakv);
          metakv[LABEL_TAG] = edge_label_name;
          metakv[basic_loader_t::SRC_LABEL_ID] =
              std::to_string(vertex_label_to_index_.at(src_label_name));
          metakv[basic_loader_t::DST_LABEL_ID] =
              std::to_string(vertex_label_to_index_.at(dst_label_name));
          meta = std::make_shared<arrow::KeyValueMetadata>();
          for (auto const& kv : metakv) {
            meta->Append(kv.first, kv.second);
          }
#else
          CHECK_ARROW_ERROR(meta->Set(LABEL_TAG, edge_label_name));
          CHECK_ARROW_ERROR(meta->Set(
              basic_loader_t::SRC_LABEL_ID,
              std::to_string(vertex_label_to_index_.at(src_label_name))));
          CHECK_ARROW_ERROR(meta->Set(
              basic_loader_t::DST_LABEL_ID,
              std::to_string(vertex_label_to_index_.at(dst_label_name))));
#endif

          tables[label_id].emplace_back(
              normalized_table->ReplaceSchemaMetadata(meta));
          edge_vertex_label_[edge_label_name].insert(
              std::make_pair(src_label_name, dst_label_name));
          edge_label_to_index_[edge_label_name] = label_id;
        }
      }
    } catch (std::exception& e) {
      RETURN_GS_ERROR(ErrorCode::kIOError, std::string(e.what()));
    }
    return tables;
  }

  boost::leaf::result<
      std::pair<std::vector<std::shared_ptr<arrow::Table>>,
                std::vector<std::vector<std::shared_ptr<arrow::Table>>>>>
  loadEVTablesFromEFiles(const std::vector<std::string>& efiles, int index,
                         int total_parts) {
    std::vector<std::string> vertex_label_names;
    {
      std::set<std::string> vertex_label_name_set;

      // We don't open file, just get metadata from filename
      for (auto& efile : efiles) {
        std::vector<std::string> sub_label_files;
        // for each type of edge, efile is separated by ;
        boost::split(sub_label_files, efile, boost::is_any_of(";"));

        for (auto& sub_efile : sub_label_files) {
          std::unique_ptr<vineyard::LocalIOAdaptor,
                          std::function<void(vineyard::LocalIOAdaptor*)>>
              io_adaptor(
                  new vineyard::LocalIOAdaptor(sub_efile + "#header_row=true"),
                  io_deleter_);
          auto meta = io_adaptor->GetMeta();
          auto src_label_name = meta.find(SRC_LABEL_TAG);
          auto dst_label_name = meta.find(DST_LABEL_TAG);

          if (src_label_name == meta.end() || dst_label_name == meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain label name");
          } else {
            vertex_label_name_set.insert(src_label_name->second);
            vertex_label_name_set.insert(dst_label_name->second);
          }
        }
      }

      vertex_label_num_ = vertex_label_name_set.size();
      vertex_label_names.resize(vertex_label_num_);
      // number label id
      label_id_t v_label_id = 0;
      for (auto& vertex_name : vertex_label_name_set) {
        vertex_label_to_index_[vertex_name] = v_label_id;
        vertex_label_names[v_label_id] = vertex_name;
        v_label_id++;
      }
    }

    std::vector<std::vector<std::shared_ptr<arrow::Table>>> etables(
        edge_label_num_);
    std::vector<OidSet<oid_t>> oids(vertex_label_num_);

    try {
      for (label_id_t e_label_id = 0; e_label_id < edge_label_num_;
           ++e_label_id) {
        std::vector<std::string> sub_label_files;
        boost::split(sub_label_files, efiles[e_label_id],
                     boost::is_any_of(";"));

        for (auto& sub_efile : sub_label_files) {
          std::unique_ptr<vineyard::LocalIOAdaptor,
                          std::function<void(vineyard::LocalIOAdaptor*)>>
              io_adaptor(
                  new vineyard::LocalIOAdaptor(sub_efile + "#header_row=true"),
                  io_deleter_);

          auto read_procedure =
              [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
            VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
            VY_OK_OR_RAISE(io_adaptor->Open());
            std::shared_ptr<arrow::Table> table;
            VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
            return table;
          };

          BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

          auto sync_schema_procedure =
              [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
            return SyncSchema(table, comm_spec_);
          };

          BOOST_LEAF_AUTO(normalized_table,
                          sync_gs_error(comm_spec_, sync_schema_procedure));

          auto adaptor_meta = io_adaptor->GetMeta();
          auto it = adaptor_meta.find(LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain label name");
          }

          std::shared_ptr<arrow::KeyValueMetadata> meta(
              new arrow::KeyValueMetadata());
          meta->Append("type", "EDGE");
          meta->Append(basic_loader_t::SRC_COLUMN, std::to_string(src_column));
          meta->Append(basic_loader_t::DST_COLUMN, std::to_string(dst_column));
          meta->Append("sub_label_num", std::to_string(sub_label_files.size()));

          std::string edge_label_name = it->second;
          meta->Append("label", edge_label_name);

          it = adaptor_meta.find(SRC_LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain src label name");
          }
          std::string src_label_name = it->second;
          auto src_label_id = vertex_label_to_index_.at(src_label_name);

          it = adaptor_meta.find(DST_LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain dst label name");
          }
          std::string dst_label_name = it->second;
          auto dst_label_id = vertex_label_to_index_.at(dst_label_name);

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
          std::unordered_map<std::string, std::string> metakv;
          meta->ToUnorderedMap(&metakv);
          metakv[LABEL_TAG] = edge_label_name;
          metakv[basic_loader_t::SRC_LABEL_ID] = std::to_string(src_label_id);
          metakv[basic_loader_t::DST_LABEL_ID] = std::to_string(dst_label_id);
          meta = std::make_shared<arrow::KeyValueMetadata>();
          for (auto const& kv : metakv) {
            meta->Append(kv.first, kv.second);
          }
#else
          CHECK_ARROW_ERROR(meta->Set(basic_loader_t::SRC_LABEL_ID,
                                      std::to_string(src_label_id)));
          CHECK_ARROW_ERROR(meta->Set(basic_loader_t::DST_LABEL_ID,
                                      std::to_string(dst_label_id)));
#endif

          auto e_table = normalized_table->ReplaceSchemaMetadata(meta);

          etables[e_label_id].emplace_back(e_table);
          edge_vertex_label_[edge_label_name].insert(
              std::make_pair(src_label_name, dst_label_name));
          if (edge_label_to_index_.find(edge_label_name) ==
              edge_label_to_index_.end()) {
            edge_label_to_index_[edge_label_name] = e_label_id;
          } else if (edge_label_to_index_[edge_label_name] != e_label_id) {
            RETURN_GS_ERROR(
                ErrorCode::kInvalidValueError,
                "Edge label is not consistent, " + edge_label_name + ": " +
                    std::to_string(e_label_id) + " vs " +
                    std::to_string(edge_label_to_index_[edge_label_name]));
          }

          // Build oid set from etable
          BOOST_LEAF_CHECK(
              oids[src_label_id].BatchInsert(e_table->column(src_column)));
          BOOST_LEAF_CHECK(
              oids[dst_label_id].BatchInsert(e_table->column(dst_column)));
        }
      }
    } catch (std::exception& e) {
      RETURN_GS_ERROR(ErrorCode::kIOError, std::string(e.what()));
    }

    // Now, oids are ready to use
    std::vector<std::shared_ptr<arrow::Table>> vtables(vertex_label_num_);

    for (auto v_label_id = 0; v_label_id < vertex_label_num_; v_label_id++) {
      auto label_name = vertex_label_names[v_label_id];
      std::vector<std::shared_ptr<arrow::Field>> schema_vector{arrow::field(
          label_name, vineyard::ConvertToArrowType<oid_t>::TypeValue())};
      BOOST_LEAF_AUTO(oid_array, oids[v_label_id].ToArrowArray());
      std::vector<std::shared_ptr<arrow::Array>> arrays{oid_array};
      auto schema = std::make_shared<arrow::Schema>(schema_vector);
      auto v_table = arrow::Table::Make(schema, arrays);
      std::shared_ptr<arrow::KeyValueMetadata> meta(
          new arrow::KeyValueMetadata());

      meta->Append("type", "VERTEX");
      meta->Append("label_index", std::to_string(v_label_id));
      meta->Append("label", label_name);
      meta->Append(basic_loader_t::ID_COLUMN, std::to_string(id_column));
      vtables[v_label_id] = v_table->ReplaceSchemaMetadata(meta);
    }
    return std::make_pair(vtables, etables);
  }

  Status readTableFromVineyard(vineyard::Client& client,
                               const ObjectID object_id,
                               std::shared_ptr<arrow::Table>& table) {
    auto pstream = client.GetObject<vineyard::ParallelStream>(object_id);
    RETURN_ON_ASSERT(pstream != nullptr,
                     "Object not exists: " + VYObjectIDToString(object_id));
    auto local_streams = pstream->GetLocalStreams<vineyard::DataframeStream>();
    int local_worker_num = comm_spec_.local_num();

    size_t split_size = local_streams.size() / local_worker_num +
                        (local_streams.size() % local_worker_num == 0 ? 0 : 1);
    int start_to_read = comm_spec_.local_id() * split_size;
    int end_to_read = std::max(local_streams.size(),
                               (comm_spec_.local_id() + 1) * split_size);

    std::mutex mutex_for_results;
    std::vector<std::shared_ptr<arrow::Table>> tables;

    auto reader = [&client, &local_streams, &mutex_for_results,
                   &tables](size_t idx) {
      // use a local client, since reading from stream may block the client.
      Client local_client;
      RETURN_ON_ERROR(local_client.Connect(client.IPCSocket()));
      auto reader = local_streams[idx]->OpenReader(local_client);
      std::shared_ptr<arrow::Table> table;
      RETURN_ON_ERROR(reader->ReadTable(table));
      VLOG(10) << "table from stream: " << table->schema()->ToString();
      {
        std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
        tables.emplace_back(table);
      }
      return Status::OK();
    };

    ThreadGroup tg;
    for (size_t idx = start_to_read; idx != end_to_read; ++idx) {
      tg.AddTask(reader, idx);
    }
    auto readers_status = tg.TakeResults();
    for (auto const& status : readers_status) {
      RETURN_ON_ERROR(status);
    }
    RETURN_ON_ASSERT(!tables.empty(),
                     "This worker doesn't receive any streams");
    table = vineyard::ConcatenateTables(tables);
    return Status::OK();
  }

  Status readRecordBatchesFromVineyard(
      vineyard::Client& client, const ObjectID object_id,
      std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
    auto pstream = client.GetObject<vineyard::ParallelStream>(object_id);
    RETURN_ON_ASSERT(pstream != nullptr,
                     "Object not exists: " + VYObjectIDToString(object_id));
    auto local_streams = pstream->GetLocalStreams<vineyard::DataframeStream>();
    int local_worker_num = comm_spec_.local_num();

    size_t split_size = local_streams.size() / local_worker_num +
                        (local_streams.size() % local_worker_num == 0 ? 0 : 1);
    size_t start_to_read = comm_spec_.local_id() * split_size;
    size_t end_to_read = std::max(local_streams.size(),
                                  (comm_spec_.local_id() + 1) * split_size);

    std::mutex mutex_for_results;

    auto reader = [&client, &local_streams, &mutex_for_results,
                   &batches](size_t idx) {
      // use a local client, since reading from stream may block the client.
      Client local_client;
      RETURN_ON_ERROR(local_client.Connect(client.IPCSocket()));
      auto reader = local_streams[idx]->OpenReader(local_client);
      std::vector<std::shared_ptr<arrow::RecordBatch>> read_batches;
      RETURN_ON_ERROR(reader->ReadRecordBatches(read_batches));
      {
        std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
        for (auto const& batch : read_batches) {
          VLOG(10) << "recordbatch from stream: "
                   << batch->schema()->ToString();
          batches.emplace_back(batch);
        }
      }
      return Status::OK();
    };

    ThreadGroup tg;
    for (size_t idx = start_to_read; idx != end_to_read; ++idx) {
      tg.AddTask(reader, idx);
    }
    auto readers_status = tg.TakeResults();
    for (auto const& status : readers_status) {
      RETURN_ON_ERROR(status);
    }
    RETURN_ON_ASSERT(batches.size() > 0,
                     "This worker doesn't receive any streams");
    return Status::OK();
  }

  Status rebuildVTableMetadata(label_id_t label_id,
                               std::shared_ptr<arrow::Table> table,
                               std::shared_ptr<arrow::Table>& target) {
    std::shared_ptr<arrow::KeyValueMetadata> meta;
    if (table->schema()->metadata() != nullptr) {
      meta = table->schema()->metadata()->Copy();
    } else {
      meta = std::make_shared<arrow::KeyValueMetadata>();
    }
    meta->Append("type", "VERTEX");
    meta->Append(basic_loader_t::ID_COLUMN, std::to_string(id_column));

    int label_meta_index = meta->FindKey(LABEL_TAG);
    RETURN_ON_ASSERT(
        label_meta_index != -1,
        "Metadata of input vertex files should contain label name");
    vertex_label_to_index_[meta->value(label_meta_index)] = label_id;

    target = table->ReplaceSchemaMetadata(meta);
    return Status::OK();
  }

  Status rebuildETableMetadata(label_id_t label_id, size_t const subtable_size,
                               std::shared_ptr<arrow::Table> table,
                               std::shared_ptr<arrow::Table>& target) {
    std::shared_ptr<arrow::KeyValueMetadata> meta;
    if (table->schema()->metadata() != nullptr) {
      meta = table->schema()->metadata()->Copy();
    } else {
      meta.reset(new arrow::KeyValueMetadata());
    }
    meta->Append("type", "EDGE");
    meta->Append(basic_loader_t::SRC_COLUMN, std::to_string(src_column));
    meta->Append(basic_loader_t::DST_COLUMN, std::to_string(dst_column));
    meta->Append("sub_label_num", std::to_string(subtable_size));

    int label_meta_index = meta->FindKey(LABEL_TAG);
    RETURN_ON_ASSERT(label_meta_index != -1,
                     "Metadata of input edge files should contain label name");
    std::string edge_label_name = meta->value(label_meta_index);

    int src_label_meta_index = meta->FindKey(SRC_LABEL_TAG);
    RETURN_ON_ASSERT(
        src_label_meta_index != -1,
        "Metadata of input edge files should contain src_label name");
    std::string src_label_name = meta->value(src_label_meta_index);

    int dst_label_meta_index = meta->FindKey(DST_LABEL_TAG);
    RETURN_ON_ASSERT(
        dst_label_meta_index != -1,
        "Metadata of input edge files should contain dst_label name");
    std::string dst_label_name = meta->value(dst_label_meta_index);

#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    std::unordered_map<std::string, std::string> metakv;
    meta->ToUnorderedMap(&metakv);
    metakv[LABEL_TAG] = edge_label_name;
    metakv[basic_loader_t::SRC_LABEL_ID] =
        std::to_string(vertex_label_to_index_.at(src_label_name));
    metakv[basic_loader_t::DST_LABEL_ID] =
        std::to_string(vertex_label_to_index_.at(dst_label_name));
    meta = std::make_shared<arrow::KeyValueMetadata>();
    for (auto const& kv : metakv) {
      meta->Append(kv.first, kv.second);
    }
#else
    RETURN_ON_ARROW_ERROR(
        meta->Set(basic_loader_t::SRC_LABEL_ID,
                  std::to_string(vertex_label_to_index_.at(src_label_name))));
    RETURN_ON_ARROW_ERROR(
        meta->Set(basic_loader_t::DST_LABEL_ID,
                  std::to_string(vertex_label_to_index_.at(dst_label_name))));
#endif

    edge_vertex_label_[edge_label_name].insert(
        std::make_pair(src_label_name, dst_label_name));
    edge_label_to_index_[edge_label_name] = label_id;

    target = table->ReplaceSchemaMetadata(meta);
    return Status::OK();
  }

  boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> gatherVTables(
      vineyard::Client& client, const std::vector<ObjectID>& vstreams) {
    using batch_group_t =
        std::unordered_map<std::string,
                           std::vector<std::shared_ptr<arrow::RecordBatch>>>;
    std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
    std::mutex mutex_for_results;
    auto reader = [this, &client, &mutex_for_results,
                   &record_batches](ObjectID const vstream) {
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      auto status = readRecordBatchesFromVineyard(client, vstream, batches);
      if (status.ok()) {
        std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
        record_batches.insert(record_batches.end(), batches.begin(),
                              batches.end());
      } else {
        LOG(ERROR) << "Failed to read from stream "
                   << VYObjectIDToString(vstream) << ": " << status.ToString();
      }
      return Status::OK();
    };

    ThreadGroup tg;
    for (auto const& vstream : vstreams) {
      tg.AddTask(reader, vstream);
    }
    tg.TakeResults();

    batch_group_t grouped_batches;
    for (auto const& batch : record_batches) {
      auto metadata = batch->schema()->metadata();
      if (metadata == nullptr) {
        LOG(ERROR) << "Invalid batch ignored: no metadata: "
                   << batch->schema()->ToString();
        continue;
      }
      std::unordered_map<std::string, std::string> meta_map;
      metadata->ToUnorderedMap(&meta_map);
      grouped_batches[meta_map.at("label")].emplace_back(batch);
    }

    std::vector<std::shared_ptr<arrow::Table>> tables;
    label_id_t v_label_id = 0;
    for (auto const& group : grouped_batches) {
      std::shared_ptr<arrow::Table> table;
      VY_OK_OR_RAISE(vineyard::RecordBatchesToTable(group.second, &table));
      std::shared_ptr<arrow::Table> result_table;
      VY_OK_OR_RAISE(rebuildVTableMetadata(v_label_id++, table, result_table));
      tables.emplace_back(result_table);
    }
    return tables;
  }

  boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
  gatherETables(vineyard::Client& client,
                const std::vector<std::vector<ObjectID>>& estreams) {
    using batch_group_t = std::unordered_map<
        std::string,
        std::map<std::pair<std::string, std::string>,
                 std::vector<std::shared_ptr<arrow::RecordBatch>>>>;
    std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
    std::mutex mutex_for_results;
    auto reader = [this, &client, &mutex_for_results,
                   &record_batches](ObjectID const estream) {
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      auto status = readRecordBatchesFromVineyard(client, estream, batches);
      if (status.ok()) {
        std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
        record_batches.insert(record_batches.end(), batches.begin(),
                              batches.end());
      } else {
        LOG(ERROR) << "Failed to read from stream "
                   << VYObjectIDToString(estream) << ": " << status.ToString();
      }
      return Status::OK();
    };

    ThreadGroup tg;
    for (auto const& esubstreams : estreams) {
      for (auto const& estream : esubstreams) {
        tg.AddTask(reader, estream);
      }
    }
    tg.TakeResults();

    batch_group_t grouped_batches;
    for (auto const& batch : record_batches) {
      auto metadata = batch->schema()->metadata();
      if (metadata == nullptr) {
        LOG(ERROR) << "Invalid batch ignored: no metadata: "
                   << batch->schema()->ToString();
        continue;
      }
      std::unordered_map<std::string, std::string> meta_map;
      metadata->ToUnorderedMap(&meta_map);
      grouped_batches[meta_map.at("label")]
                     [std::make_pair(meta_map.at("src_label"),
                                     meta_map.at("dst_label"))]
                         .emplace_back(batch);
    }

    std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables;
    label_id_t e_label_id = 0;
    for (auto const& group : grouped_batches) {
      std::shared_ptr<arrow::Table> table;
      std::vector<std::shared_ptr<arrow::Table>> subtables;
      for (auto const& subgroup : group.second) {
        VY_OK_OR_RAISE(vineyard::RecordBatchesToTable(subgroup.second, &table));
        std::shared_ptr<arrow::Table> result_table;
        VY_OK_OR_RAISE(rebuildETableMetadata(e_label_id, group.second.size(),
                                             table, result_table));
        subtables.emplace_back(result_table);
      }
      e_label_id += 1;
      tables.emplace_back(subtables);
    }
    return tables;
  }

  arrow::Status swapColumn(std::shared_ptr<arrow::Table> in, int lhs_index,
                           int rhs_index, std::shared_ptr<arrow::Table>* out) {
    if (lhs_index == rhs_index) {
      out = &in;
      return arrow::Status::OK();
    }
    if (lhs_index > rhs_index) {
      return arrow::Status::Invalid("lhs index must smaller than rhs index.");
    }
    auto lhs_field = in->schema()->field(lhs_index);
    auto lhs_column = in->column(lhs_index);
    auto rhs_field = in->schema()->field(rhs_index);
    auto rhs_column = in->column(rhs_index);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(in->RemoveColumn(rhs_index, &in));
    CHECK_ARROW_ERROR(in->RemoveColumn(lhs_index, &in));
    CHECK_ARROW_ERROR(in->AddColumn(lhs_index, rhs_field, rhs_column, out));
    CHECK_ARROW_ERROR(in->AddColumn(rhs_index, lhs_field, lhs_column, out));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(in, in->RemoveColumn(rhs_index));
    CHECK_ARROW_ERROR_AND_ASSIGN(in, in->RemoveColumn(lhs_index));
    CHECK_ARROW_ERROR_AND_ASSIGN(
        *out, in->AddColumn(lhs_index, rhs_field, rhs_column));
    CHECK_ARROW_ERROR_AND_ASSIGN(
        *out, in->AddColumn(rhs_index, lhs_field, lhs_column));
#endif
    return arrow::Status::OK();
  }

  std::map<std::string, label_id_t> vertex_label_to_index_;
  std::map<std::string, label_id_t> edge_label_to_index_;
  std::map<std::string, std::set<std::pair<std::string, std::string>>>
      edge_vertex_label_;

  vineyard::Client& client_;
  grape::CommSpec comm_spec_;
  std::vector<std::string> efiles_, vfiles_;
  bool possible_duplicate_oid = false;

  label_id_t vertex_label_num_, edge_label_num_;
  std::vector<vineyard::ObjectID> v_streams_;
  std::vector<std::vector<vineyard::ObjectID>> e_streams_;
  std::vector<std::shared_ptr<arrow::Table>> partial_v_tables_;
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables_;
  partitioner_t partitioner_;

  bool directed_;
  bool generate_eid_;
  basic_loader_t basic_arrow_fragment_loader_;
  std::function<void(vineyard::LocalIOAdaptor*)> io_deleter_ =
      [](vineyard::LocalIOAdaptor* adaptor) {
        VINEYARD_CHECK_OK(adaptor->Close());
        delete adaptor;
      };
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
