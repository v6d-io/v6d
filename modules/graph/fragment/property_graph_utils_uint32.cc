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

#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils_impl.h"

namespace vineyard {

template boost::leaf::result<void> generate_outer_vertices_map<uint32_t>(
    const IdParser<uint32_t>& parser, fid_t fid,
    property_graph_types::LABEL_ID_TYPE vertex_label_num,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> dsts,
    const std::vector<uint32_t>& start_ids,
    std::vector<ska::flat_hash_map<
        uint32_t, uint32_t, typename Hashmap<uint32_t, uint32_t>::KeyHash>>&
        ovg2l_maps,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>>& ovgid_lists);

template boost::leaf::result<void> generate_local_id_list<uint32_t>(
    IdParser<uint32_t>& parser, std::shared_ptr<arrow::ChunkedArray>&& gid_list,
    fid_t fid,
    const std::vector<ska::flat_hash_map<
        uint32_t, uint32_t, typename Hashmap<uint32_t, uint32_t>::KeyHash>>&
        ovg2l_maps,
    int concurrency,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>>& lid_list,
    arrow::MemoryPool* pool);

template void sort_edges_with_respect_to_vertex<uint32_t, uint64_t>(
    vineyard::PodArrayBuilder<
        property_graph_utils::NbrUnit<uint32_t, uint64_t>>& builder,
    const int64_t* offsets, uint32_t tvnum, int concurrency);

template void check_is_multigraph<uint32_t, uint64_t>(
    vineyard::PodArrayBuilder<
        property_graph_utils::NbrUnit<uint32_t, uint64_t>>& builder,
    const int64_t* offsets, uint32_t tvnum, int concurrency,
    bool& is_multigraph);

template boost::leaf::result<void> generate_directed_csr<uint32_t, uint64_t>(
    Client& client, IdParser<uint32_t>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>> dst_chunks,
    std::vector<uint32_t> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>&
        edges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& edge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void> generate_directed_csc<uint32_t, uint64_t>(
    Client& client, IdParser<uint32_t>& parser, std::vector<uint32_t> tvnums,
    int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>&
        oedges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& oedge_offsets,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>&
        iedges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& iedge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void> generate_undirected_csr<uint32_t, uint64_t>(
    Client& client, IdParser<uint32_t>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>> dst_chunks,
    std::vector<uint32_t> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>&
        edges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& edge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void>
generate_undirected_csr_memopt<uint32_t, uint64_t>(
    Client& client, IdParser<uint32_t>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<uint32_t>>> dst_chunks,
    std::vector<uint32_t> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>&
        edges,
    std::vector<std::shared_ptr<FixedInt64Builder>>& edge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void> varint_encoding_edges<uint32_t, uint64_t>(
    Client& client, const bool directed,
    const property_graph_types::LABEL_ID_TYPE vertex_label_num,
    const property_graph_types::LABEL_ID_TYPE edge_label_num,
    std::vector<std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>>&
        ie_lists,
    std::vector<std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint32_t, uint64_t>>>>>&
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
    const int concurrency);

}  // namespace vineyard
