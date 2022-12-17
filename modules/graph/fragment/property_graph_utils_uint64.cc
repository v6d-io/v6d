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

#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils_impl.h"

namespace vineyard {

template boost::leaf::result<void> generate_outer_vertices_map<uint64_t>(
    const IdParser<uint64_t>& parser, fid_t fid,
    property_graph_types::LABEL_ID_TYPE vertex_label_num,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> dsts,
    const std::vector<uint64_t>& start_ids,
    std::vector<ska::flat_hash_map<
        uint64_t, uint64_t, typename Hashmap<uint64_t, uint64_t>::KeyHash>>&
        ovg2l_maps,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>>& ovgid_lists);

template boost::leaf::result<void> generate_local_id_list<uint64_t>(
    IdParser<uint64_t>& parser, std::shared_ptr<arrow::ChunkedArray>&& gid_list,
    fid_t fid,
    const std::vector<ska::flat_hash_map<
        uint64_t, uint64_t, typename Hashmap<uint64_t, uint64_t>::KeyHash>>&
        ovg2l_maps,
    int concurrency,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>>& lid_list,
    arrow::MemoryPool* pool);

template void sort_edges_with_respect_to_vertex<uint64_t, uint64_t>(
    vineyard::PodArrayBuilder<
        property_graph_utils::NbrUnit<uint64_t, uint64_t>>& builder,
    std::shared_ptr<arrow::Int64Array> offsets, uint64_t tvnum,
    int concurrency);

template void check_is_multigraph<uint64_t, uint64_t>(
    vineyard::PodArrayBuilder<
        property_graph_utils::NbrUnit<uint64_t, uint64_t>>& builder,
    std::shared_ptr<arrow::Int64Array> offsets, uint64_t tvnum, int concurrency,
    bool& is_multigraph);

template boost::leaf::result<void> generate_directed_csr<uint64_t, uint64_t>(
    Client& client, IdParser<uint64_t>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>> dst_chunks,
    std::vector<uint64_t> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint64_t, uint64_t>>>>&
        edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void> generate_directed_csc<uint64_t, uint64_t>(
    Client& client, IdParser<uint64_t>& parser, std::vector<uint64_t> tvnums,
    int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint64_t, uint64_t>>>>&
        oedges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& oedge_offsets,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint64_t, uint64_t>>>>&
        iedges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& iedge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void> generate_undirected_csr<uint64_t, uint64_t>(
    Client& client, IdParser<uint64_t>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>> dst_chunks,
    std::vector<uint64_t> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint64_t, uint64_t>>>>&
        edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    bool& is_multigraph);

template boost::leaf::result<void>
generate_undirected_csr_memopt<uint64_t, uint64_t>(
    Client& client, IdParser<uint64_t>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<uint64_t>>> dst_chunks,
    std::vector<uint64_t> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<uint64_t, uint64_t>>>>&
        edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    bool& is_multigraph);

}  // namespace vineyard
