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

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/worker/comm_spec.h"

#include "basic/ds/dataframe.h"
#include "basic/ds/tensor.h"
#include "client/client.h"
#include "io/io/io_factory.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/loader/basic_ev_fragment_loader.h"
#include "graph/utils/partitioner.h"
#include "graph/vertex_map/arrow_vertex_map.h"

#define HASH_PARTITION

namespace vineyard {

class DataframeStream;
class RecordBatchStream;
class ParallelStream;

Status ReadRecordBatchesFromVineyardStream(
    Client& client, std::shared_ptr<ParallelStream>& pstream,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num);

Status ReadRecordBatchesFromVineyardDataFrame(
    Client& client, std::shared_ptr<GlobalDataFrame>& gdf,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num);

Status ReadRecordBatchesFromVineyard(
    Client& client, const ObjectID object_id,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num);

/**
 * @brief When the stream is empty, the result `table` will be set as nullptr.
 */
Status ReadTableFromVineyardStream(Client& client,
                                   std::shared_ptr<ParallelStream>& pstream,
                                   std::shared_ptr<arrow::Table>& table,
                                   int part_id, int part_num);

/**
 * @brief When no local chunk, the result `table` will be set as nullptr.
 */
Status ReadTableFromVineyardDataFrame(Client& client,
                                      std::shared_ptr<GlobalDataFrame>& gdf,
                                      std::shared_ptr<arrow::Table>& table,
                                      int part_id, int part_num);

/**
 * @brief The result `table` will be set as nullptr.
 */
Status ReadTableFromVineyard(Client& client, const ObjectID object_id,
                             std::shared_ptr<arrow::Table>& table, int part_id,
                             int part_num);

Status ReadTableFromPandas(const std::string& data,
                           std::shared_ptr<arrow::Table>& table);

Status ReadTableFromLocation(const std::string& location,
                             std::shared_ptr<arrow::Table>& table, int index,
                             int total_parts);

/** Note [GatherETables and GatherVTables]
 *
 * GatherETables and GatherVTables gathers all edges and vertices as table from
 * multiple streams.
 *
 * It requires (one of the follows):
 *
 * + all chunks in the stream has a "label" (and "src_label", "dst_label" for
 *   edges) in meta, and at least one batch available on each worker.
 *
 * + or all chunks doesn't have such meta.
 */

boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
GatherETables(Client& client,
              const std::vector<std::vector<ObjectID>>& estreams, int part_id,
              int part_num);

boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> GatherVTables(
    Client& client, const std::vector<ObjectID>& vstreams, int part_id,
    int part_num);

class DataLoader {
 public:
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using table_vec_t = std::vector<std::shared_ptr<arrow::Table>>;

 protected:
  // These consts represent the key in the path of vfile/efile
  static constexpr const char* LABEL_TAG = "label";
  static constexpr const char* SRC_LABEL_TAG = "src_label";
  static constexpr const char* DST_LABEL_TAG = "dst_label";
  static constexpr const char* CONSOLIDATE_TAG = "consolidate";
  static constexpr const char* MARKER = "PROGRESS--GRAPH-LOADING-";

  static constexpr int id_column = 0;

  using vertex_table_info_t =
      std::map<std::string, std::shared_ptr<arrow::Table>>;
  using edge_table_info_t = std::vector<InputTable>;

 public:
  /**
   *
   * @param client
   * @param comm_spec
   * @param efiles An example of efile:
   * /data/twitter_e_0_0_0#src_label=v0&dst_label=v0&label=e0;/data/twitter_e_0_1_0#src_label=v0&dst_label=v1&label=e0;/data/twitter_e_1_0_0#src_label=v1&dst_label=v0&label=e0;/data/twitter_e_1_1_0#src_label=v1&dst_label=v1&label=e0
   * @param vfiles An example of vfile: /data/twitter_v_0#label=v0
   */
  DataLoader(Client& client, const grape::CommSpec& comm_spec,
             const std::vector<std::string>& efiles,
             const std::vector<std::string>& vfiles)
      : client_(client),
        comm_spec_(comm_spec),
        efiles_(efiles),
        vfiles_(vfiles) {}

  DataLoader(Client& client, const grape::CommSpec& comm_spec,
             const std::vector<std::string>& efiles)
      : client_(client), comm_spec_(comm_spec), efiles_(efiles), vfiles_() {}

  DataLoader(Client& client, const grape::CommSpec& comm_spec,
             std::vector<std::shared_ptr<arrow::Table>> const& partial_v_tables,
             std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
                 partial_e_tables)
      : client_(client),
        comm_spec_(comm_spec),
        partial_v_tables_(partial_v_tables),
        partial_e_tables_(partial_e_tables) {}

  DataLoader(Client& client, const grape::CommSpec& comm_spec,
             std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
                 partial_e_tables)
      : client_(client),
        comm_spec_(comm_spec),
        partial_v_tables_(),
        partial_e_tables_(partial_e_tables) {}

  ~DataLoader() = default;

  boost::leaf::result<std::pair<table_vec_t, std::vector<table_vec_t>>>
  LoadVertexEdgeTables();

  boost::leaf::result<table_vec_t> LoadVertexTables();

  boost::leaf::result<std::vector<table_vec_t>> LoadEdgeTables();

 protected:  // for subclasses
  boost::leaf::result<vineyard::ObjectID> resolveVineyardObject(
      std::string const& source);

  boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
  loadVertexTables(const std::vector<std::string>& files, int index,
                   int total_parts);

  boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
  loadEdgeTables(const std::vector<std::string>& files, int index,
                 int total_parts);

  /// Do some necessary sanity checks.
  boost::leaf::result<void> sanityChecks(std::shared_ptr<arrow::Table> table);

  Client& client_;
  grape::CommSpec comm_spec_;
  std::vector<std::string> efiles_, vfiles_;

  std::vector<ObjectID> v_streams_;
  std::vector<std::vector<ObjectID>> e_streams_;
  // partial means every worker has its own part of the table
  std::vector<std::shared_ptr<arrow::Table>> partial_v_tables_;
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables_;

  std::function<void(IIOAdaptor*)> io_deleter_ = [](IIOAdaptor* adaptor) {
    VINEYARD_DISCARD(adaptor->Close());
    delete adaptor;
  };
};

template <typename OID_T = property_graph_types::OID_TYPE,
          typename VID_T = property_graph_types::VID_TYPE>
class ArrowFragmentLoader : public DataLoader {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = ArrowArrayType<oid_t>;
  using vid_array_t = ArrowArrayType<vid_t>;
  using vertex_map_t = ArrowVertexMap<internal_oid_t, vid_t>;
  using local_vertex_map_t = ArrowLocalVertexMap<internal_oid_t, vid_t>;

  using table_vec_t = std::vector<std::shared_ptr<arrow::Table>>;
  using oid_array_vec_t = std::vector<std::shared_ptr<oid_array_t>>;
  using vid_array_vec_t = std::vector<std::shared_ptr<vid_array_t>>;

#ifdef HASH_PARTITION
  using partitioner_t = HashPartitioner<oid_t>;
#else
  using partitioner_t = SegmentedPartitioner<oid_t>;
#endif

  using basic_fragment_loader_t =
      BasicEVFragmentLoader<OID_T, VID_T, partitioner_t>;

 protected:
  // These consts represent the key in the path of vfile/efile
  static constexpr const char* LABEL_TAG = "label";
  static constexpr const char* SRC_LABEL_TAG = "src_label";
  static constexpr const char* DST_LABEL_TAG = "dst_label";
  static constexpr const char* CONSOLIDATE_TAG = "consolidate";
  static constexpr const char* MARKER = "PROGRESS--GRAPH-LOADING-";

  static constexpr int id_column = 0;

  using vertex_table_info_t =
      std::map<std::string, std::shared_ptr<arrow::Table>>;
  using edge_table_info_t = std::vector<InputTable>;

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
  ArrowFragmentLoader(Client& client, const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      const std::vector<std::string>& vfiles,
                      bool directed = true, bool generate_eid = false,
                      bool retain_oid = false, bool local_vertex_map = false,
                      bool compact_edges = false, bool use_perfect_hash = false)
      : DataLoader(client, comm_spec, efiles, vfiles),
        client_(client),
        comm_spec_(comm_spec),
        directed_(directed),
        generate_eid_(generate_eid),
        retain_oid_(retain_oid),
        local_vertex_map_(local_vertex_map),
        compact_edges_(compact_edges),
        use_perfect_hash_(use_perfect_hash) {}

  ArrowFragmentLoader(Client& client, const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      bool directed = true, bool generate_eid = false,
                      bool retain_oid = false, bool local_vertex_map = false,
                      bool compact_edges = false, bool use_perfect_hash = false)
      : DataLoader(client, comm_spec, efiles),
        client_(client),
        comm_spec_(comm_spec),
        directed_(directed),
        generate_eid_(generate_eid),
        retain_oid_(retain_oid),
        local_vertex_map_(local_vertex_map),
        compact_edges_(compact_edges),
        use_perfect_hash_(use_perfect_hash) {}

  ArrowFragmentLoader(
      Client& client, const grape::CommSpec& comm_spec,
      std::vector<std::shared_ptr<arrow::Table>> const& partial_v_tables,
      std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
          partial_e_tables,
      bool directed = true, bool generate_eid = false, bool retain_oid = false,
      bool local_vertex_map = false, bool compact_edges = false,
      bool use_perfect_hash = false)
      : DataLoader(client, comm_spec, partial_v_tables, partial_e_tables),
        client_(client),
        comm_spec_(comm_spec),
        directed_(directed),
        generate_eid_(generate_eid),
        retain_oid_(retain_oid),
        local_vertex_map_(local_vertex_map),
        compact_edges_(compact_edges),
        use_perfect_hash_(use_perfect_hash) {}

  ArrowFragmentLoader(
      Client& client, const grape::CommSpec& comm_spec,
      std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
          partial_e_tables,
      bool directed = true, bool generate_eid = false, bool retain_oid = false,
      bool local_vertex_map = false, bool compact_edges = false,
      bool use_perfect_hash = false)
      : DataLoader(client, comm_spec, partial_e_tables),
        client_(client),
        comm_spec_(comm_spec),
        directed_(directed),
        generate_eid_(generate_eid),
        retain_oid_(retain_oid),
        local_vertex_map_(local_vertex_map),
        compact_edges_(compact_edges),
        use_perfect_hash_(use_perfect_hash) {}

  ~ArrowFragmentLoader() = default;

  boost::leaf::result<ObjectID> LoadFragment();

  boost::leaf::result<ObjectID> LoadFragment(
      const std::vector<std::string>& efiles,
      const std::vector<std::string>& vfiles);

  boost::leaf::result<ObjectID> LoadFragment(
      std::pair<table_vec_t, std::vector<table_vec_t>> raw_v_e_tables);

  boost::leaf::result<ObjectID> LoadFragmentAsFragmentGroup();

  boost::leaf::result<ObjectID> LoadFragmentAsFragmentGroup(
      const std::vector<std::string>& efiles,
      const std::vector<std::string>& vfiles);

  boost::leaf::result<vineyard::ObjectID> AddLabelsToFragment(
      vineyard::ObjectID frag_id);

  boost::leaf::result<vineyard::ObjectID> AddLabelsToFragmentAsFragmentGroup(
      vineyard::ObjectID frag_id);

  boost::leaf::result<vineyard::ObjectID> AddDataToExistedVLabel(
      vineyard::ObjectID frag_id, PropertyGraphSchema::LabelId label_id);

  boost::leaf::result<vineyard::ObjectID> addDataToExistedVLabel(
      vineyard::ObjectID frag_id, PropertyGraphSchema::LabelId label_id,
      std::pair<table_vec_t, std::vector<table_vec_t>> raw_v_e_tables);

  boost::leaf::result<vineyard::ObjectID> AddDataToExistedELabel(
      vineyard::ObjectID frag_id, PropertyGraphSchema::LabelId label_id);

  boost::leaf::result<vineyard::ObjectID> addDataToExistedELabel(
      vineyard::ObjectID frag_id, PropertyGraphSchema::LabelId label_id,
      std::pair<table_vec_t, std::vector<table_vec_t>> raw_v_e_tables);

  using DataLoader::LoadEdgeTables;
  using DataLoader::LoadVertexEdgeTables;
  using DataLoader::LoadVertexTables;

 protected:  // for subclasses
  boost::leaf::result<void> initPartitioner();

  boost::leaf::result<std::pair<vertex_table_info_t, edge_table_info_t>>
  preprocessInputs(
      const std::vector<std::shared_ptr<arrow::Table>>& v_tables,
      const std::vector<std::vector<std::shared_ptr<arrow::Table>>>& e_tables,
      const std::set<std::string>& previous_vertex_labels =
          std::set<std::string>());

  boost::leaf::result<vineyard::ObjectID> addVerticesAndEdges(
      vineyard::ObjectID frag_id,
      std::pair<table_vec_t, std::vector<table_vec_t>> raw_v_e_tables);

  using DataLoader::loadEdgeTables;
  using DataLoader::loadVertexTables;
  using DataLoader::resolveVineyardObject;
  using DataLoader::sanityChecks;

  Client& client_;
  grape::CommSpec comm_spec_;

  partitioner_t partitioner_;

  bool directed_ = true;
  bool generate_eid_ = false;
  bool retain_oid_ = false;
  bool local_vertex_map_ = false;
  bool compact_edges_ = false;
  bool use_perfect_hash_ = false;

  std::function<void(IIOAdaptor*)> io_deleter_ = [](IIOAdaptor* adaptor) {
    VINEYARD_DISCARD(adaptor->Close());
    delete adaptor;
  };
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
