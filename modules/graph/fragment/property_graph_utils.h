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

#ifndef MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_H_
#define MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "boost/algorithm/string.hpp"
#include "boost/leaf.hpp"

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/utils/atomic_ops.h"
#include "grape/utils/vertex_array.h"

#include "basic/ds/hashmap.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"
#include "graph/utils/mpi_utils.h"

namespace vineyard {

template <typename VID_T>
boost::leaf::result<void> generate_outer_vertices_map(
    const IdParser<VID_T>& parser, fid_t fid,
    property_graph_types::LABEL_ID_TYPE vertex_label_num,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> dsts,
    const std::vector<VID_T>& start_ids,
    std::vector<ska::flat_hash_map<
        VID_T, VID_T, typename Hashmap<VID_T, VID_T>::KeyHash>>& ovg2l_maps,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>>& ovgid_lists);

template <typename VID_T>
boost::leaf::result<void> generate_local_id_list(
    IdParser<VID_T>& parser, std::shared_ptr<arrow::ChunkedArray>&& gid_list,
    fid_t fid,
    const std::vector<ska::flat_hash_map<
        VID_T, VID_T, typename Hashmap<VID_T, VID_T>::KeyHash>>& ovg2l_maps,
    int concurrency,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>>& lid_list,
    arrow::MemoryPool* pool = arrow::default_memory_pool());

template <typename VID_T, typename EID_T>
void sort_edges_with_respect_to_vertex(
    vineyard::PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>&
        builder,
    const int64_t* offsets, VID_T tvnum, int concurrency);

template <typename VID_T, typename EID_T>
void check_is_multigraph(
    vineyard::PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>&
        builder,
    const int64_t* offsets, VID_T tvnum, int concurrency, bool& is_multigraph);

/**
 * @brief Generate CSR from given COO.
 */
template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_directed_csr(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& edge_offsets,
    bool& is_multigraph);

/**
 * @brief Generate CSC from given CSR.
 */
template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_directed_csc(
    Client& client, IdParser<VID_T>& parser, std::vector<VID_T> tvnums,
    int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& oedges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& oedge_offsets,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& iedges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& iedge_offsets,
    bool& is_multigraph);

/**
 * @brief Generate CSR and CSC from given COO, scan once, and generate both
 * direction at the same time.
 */
template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_undirected_csr(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& edge_offsets,
    bool& is_multigraph);

/**
 * @brief Generate CSR and CSC from given COO, scan twice, generate CSR from
 * COO, and then generate CSC from CSR.
 */
template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_undirected_csr_memopt(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& edge_offsets,
    bool& is_multigraph);

template <typename VID_T, typename EID_T>
boost::leaf::result<void> varint_encoding_edges(
    Client& client, const bool directed,
    const property_graph_types::LABEL_ID_TYPE vertex_label_num,
    const property_graph_types::LABEL_ID_TYPE edge_label_num,
    std::vector<std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>>&
        ie_lists,
    std::vector<std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>>&
        oe_lists,
    std::vector<std::vector<std::shared_ptr<FixedUInt8Builder>>>&
        compact_ie_lists,
    std::vector<std::vector<std::shared_ptr<FixedUInt8Builder>>>&
        compact_oe_lists,
    const std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>&
        ie_offsets_lists,
    const std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>&
        oe_offsets_lists,
    std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>&
        ie_boffsets_lists,
    std::vector<std::vector<std::shared_ptr<FixedInt64Builder>>>&
        oe_boffsets_lists,
    const int concurrency = std::thread::hardware_concurrency());
}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_H_
