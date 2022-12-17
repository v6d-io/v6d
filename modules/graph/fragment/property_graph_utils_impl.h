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

#ifndef MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_IMPL_H_
#define MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_IMPL_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"

namespace vineyard {

inline void parallel_prefix_sum(const int* input, int64_t* output,
                                size_t length, int concurrency) {
  size_t bsize =
      std::max(static_cast<size_t>(1024),
               static_cast<size_t>((length + concurrency - 1) / concurrency));
  int thread_num = static_cast<int>((length + bsize - 1) / bsize);

  auto block_prefix = [&](int i) {
    size_t begin = std::min(static_cast<size_t>(i) * bsize, length);
    size_t end = std::min(begin + bsize, length);
    output[begin] = input[begin];
    for (++begin; begin < end; ++begin) {
      output[begin] = input[begin] + output[begin - 1];
    }
  };

  std::vector<std::thread> threads_prefix;
  for (int i = 0; i < thread_num; ++i) {
    threads_prefix.emplace_back(block_prefix, i);
  }
  for (auto& thrd : threads_prefix) {
    thrd.join();
  }

  std::vector<int64_t> block_sum(thread_num);
  {
    size_t end = std::min(bsize, length);
    block_sum[0] = output[end - 1];
  }
  for (int i = 1; i < thread_num; ++i) {
    size_t begin = std::min(static_cast<size_t>(i) * bsize + bsize, length);
    block_sum[i] = block_sum[i - 1] + output[begin - 1];
  }

  auto block_add = [&](int i) {
    size_t begin = std::min(static_cast<size_t>(i) * bsize, length);
    size_t end = std::min(begin + bsize, length);
    for (; begin < end; ++begin) {
      output[begin] = output[begin] + block_sum[i - 1];
    }
  };

  std::vector<std::thread> threads_sum;
  for (int i = 1; i < thread_num; ++i) {
    threads_sum.emplace_back(block_add, i);
  }
  for (auto& thrd : threads_sum) {
    thrd.join();
  }
}

template <typename VID_T>
boost::leaf::result<void> generate_outer_vertices_map(
    const IdParser<VID_T>& parser, fid_t fid,
    property_graph_types::LABEL_ID_TYPE vertex_label_num,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> srcs,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> dsts,
    const std::vector<VID_T>& start_ids,
    std::vector<ska::flat_hash_map<
        VID_T, VID_T, typename Hashmap<VID_T, VID_T>::KeyHash>>& ovg2l_maps,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>>& ovgid_lists) {
  using vid_array_t = ArrowArrayType<VID_T>;

  ovg2l_maps.resize(vertex_label_num);
  ovgid_lists.resize(vertex_label_num);
  std::vector<std::shared_ptr<ArrowBuilderType<VID_T>>> ovgid_list_builders(
      vertex_label_num);
  for (property_graph_types::LABEL_ID_TYPE label = 0; label < vertex_label_num;
       ++label) {
    ovgid_list_builders[label] = std::make_shared<ArrowBuilderType<VID_T>>();
  }

  auto gid_array = ConcatenateChunkedArrays({srcs, dsts});
  if (gid_array != nullptr /* may be empty graph */) {
    for (auto const& chunk : gid_array->chunks()) {
      auto array = std::dynamic_pointer_cast<vid_array_t>(chunk);
      const VID_T* arr = array->raw_values();
      for (int64_t i = 0; i < array->length(); ++i) {
        if (parser.GetFid(arr[i]) != fid) {
          auto label = parser.GetLabelId(arr[i]);
          if (ovg2l_maps[label].find(arr[i]) == ovg2l_maps[label].end()) {
            // for de-dup, the value will be updated later
            ovg2l_maps[label].emplace(arr[i], -1);
            ARROW_OK_OR_RAISE(ovgid_list_builders[label]->Append(arr[i]));
          }
        }
      }
    }
  }

  for (property_graph_types::LABEL_ID_TYPE label = 0; label < vertex_label_num;
       ++label) {
    ARROW_OK_OR_RAISE(ovgid_list_builders[label]->Finish(&ovgid_lists[label]));

    // sort on the raw list
    VID_T* current_list =
        const_cast<VID_T*>(ovgid_lists[label]->raw_values());  // UNSAFE
    std::sort(current_list, current_list + ovgid_lists[label]->length());

    // update the hashmap entries
    for (int64_t k = 0; k < ovgid_lists[label]->length(); ++k) {
      ovg2l_maps[label][current_list[k]] = start_ids[label] + k;
    }
  }
  return {};
}

template <typename VID_T>
boost::leaf::result<void> generate_local_id_list(
    IdParser<VID_T>& parser, std::shared_ptr<arrow::ChunkedArray>&& gid_list,
    fid_t fid,
    const std::vector<ska::flat_hash_map<
        VID_T, VID_T, typename Hashmap<VID_T, VID_T>::KeyHash>>& ovg2l_maps,
    int concurrency,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>>& lid_list,
    arrow::MemoryPool* pool) {
  std::vector<std::shared_ptr<arrow::Array>> chunks = gid_list->chunks();
  lid_list.resize(gid_list->num_chunks());  // reserve the space
  gid_list.reset();  // release the reference of chunked arrays

  parallel_for(
      static_cast<size_t>(0), chunks.size(),
      [pool, fid, &parser, &ovg2l_maps, &chunks,
       &lid_list](size_t chunk_index) -> boost::leaf::result<void> {
        ArrowBuilderType<VID_T> builder(pool);
        auto chunk = std::dynamic_pointer_cast<ArrowArrayType<VID_T>>(
            chunks[chunk_index]);
        chunks[chunk_index].reset();  // release the used chunks
        ARROW_OK_OR_RAISE(builder.Resize(chunk->length()));

        const VID_T* vec = chunk->raw_values();
        for (int64_t i = 0; i < chunk->length(); ++i) {
          VID_T gid = vec[i];
          if (parser.GetFid(gid) == fid) {
            builder[i] = parser.GenerateId(0, parser.GetLabelId(gid),
                                           parser.GetOffset(gid));
          } else {
            builder[i] = ovg2l_maps[parser.GetLabelId(gid)].at(gid);
          }
        }
        ARROW_OK_OR_RAISE(builder.Advance(chunk->length()));
        ARROW_OK_OR_RAISE(builder.Finish(&lid_list[chunk_index]));
        return {};
      },
      concurrency);
  return {};
}

template <typename VID_T, typename EID_T>
void sort_edges_with_respect_to_vertex(
    vineyard::PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>&
        builder,
    std::shared_ptr<arrow::Int64Array> offsets, VID_T tvnum, int concurrency) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  const int64_t* offsets_ptr = offsets->raw_values();
  if (concurrency == 1) {
    for (VID_T i = 0; i < tvnum; ++i) {
      nbr_unit_t* begin = builder.MutablePointer(offsets_ptr[i]);
      nbr_unit_t* end = builder.MutablePointer(offsets_ptr[i + 1]);
      std::sort(begin, end, [](const nbr_unit_t& lhs, const nbr_unit_t& rhs) {
        return lhs.vid < rhs.vid;
      });
    }
  } else {
    parallel_for(
        static_cast<VID_T>(0), tvnum,
        [offsets_ptr, &builder](VID_T i) {
          nbr_unit_t* begin = builder.MutablePointer(offsets_ptr[i]);
          nbr_unit_t* end = builder.MutablePointer(offsets_ptr[i + 1]);
          std::sort(begin, end,
                    [](const nbr_unit_t& lhs, const nbr_unit_t& rhs) {
                      return lhs.vid < rhs.vid;
                    });
        },
        concurrency);
  }
}

template <typename VID_T, typename EID_T>
void check_is_multigraph(
    vineyard::PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>&
        builder,
    std::shared_ptr<arrow::Int64Array> offsets, VID_T tvnum, int concurrency,
    bool& is_multigraph) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;
  const int64_t* offsets_ptr = offsets->raw_values();
  if (concurrency == 1) {
    for (VID_T i = 0; i < tvnum; ++i) {
      nbr_unit_t* begin = builder.MutablePointer(offsets_ptr[i]);
      nbr_unit_t* end = builder.MutablePointer(offsets_ptr[i + 1]);
      nbr_unit_t* loc = std::adjacent_find(
          begin, end, [](const nbr_unit_t& lhs, const nbr_unit_t& rhs) {
            return lhs.vid == rhs.vid;
          });
      if (loc != end) {
        is_multigraph = true;
        break;
      }
    }
  } else {
    parallel_for(
        static_cast<VID_T>(0), tvnum,
        [offsets_ptr, &builder, &is_multigraph](VID_T i) {
          if (!is_multigraph) {
            nbr_unit_t* begin = builder.MutablePointer(offsets_ptr[i]);
            nbr_unit_t* end = builder.MutablePointer(offsets_ptr[i + 1]);
            nbr_unit_t* loc = std::adjacent_find(
                begin, end, [](const nbr_unit_t& lhs, const nbr_unit_t& rhs) {
                  return lhs.vid == rhs.vid;
                });
            if (loc != end) {
              __sync_or_and_fetch(
                  reinterpret_cast<unsigned char*>(&is_multigraph), 1);
            }
          }
        },
        concurrency);
  }
}

template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_directed_csr(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    bool& is_multigraph) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  int64_t num_chunks = src_chunks.size();
  std::vector<std::vector<int>> degree(vertex_label_num);
  std::vector<int64_t> actual_edge_num(vertex_label_num, 0);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    degree[v_label].resize(tvnums[v_label], 0);
  }

  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&degree, &parser, &src_chunks](int64_t chunk_index) {
        auto src_array = src_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();

        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          grape::atomic_add(
              degree[parser.GetLabelId(src_id)][parser.GetOffset(src_id)], 1);
        }
      },
      concurrency);

  std::vector<std::vector<int64_t>> offsets(vertex_label_num);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    auto tvnum = tvnums[v_label];
    auto& offset_vec = offsets[v_label];
    auto& degree_vec = degree[v_label];

    offset_vec.resize(tvnum + 1);
    offset_vec[0] = 0;

    if (tvnum > 0) {
      parallel_prefix_sum(degree_vec.data(), &offset_vec[1], tvnum,
                          concurrency);
    }
    // build the arrow's offset array
    std::shared_ptr<arrow::Buffer> offsets_buffer;
    ARROW_OK_ASSIGN_OR_RAISE(
        offsets_buffer, arrow::AllocateBuffer((tvnum + 1) * sizeof(int64_t),
                                              arrow::default_memory_pool()));
    memcpy(offsets_buffer->mutable_data(), offset_vec.data(),
           (tvnum + 1) * sizeof(int64_t));
    edge_offsets[v_label] = std::make_shared<arrow::Int64Array>(
        arrow::int64(), tvnum + 1, offsets_buffer, nullptr, 0, 0);
    actual_edge_num[v_label] = offset_vec[tvnum];
  }
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    edges[v_label] = std::make_shared<PodArrayBuilder<nbr_unit_t>>(
        client, actual_edge_num[v_label]);
  }

  VLOG(100) << "Start building the CSR ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);
  for (int64_t i = 0; i < num_chunks; ++i) {
    chunk_offsets[i + 1] = chunk_offsets[i] + src_chunks[i]->length();
  }
  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&src_chunks, &dst_chunks, &parser, &edges, &offsets,
       &chunk_offsets](int64_t chunk_index) {
        auto src_array = src_chunks[chunk_index];
        auto dst_array = dst_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();
        const VID_T* dst_list_ptr = dst_array->raw_values();
        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          int v_label = parser.GetLabelId(src_id);
          int64_t v_offset = parser.GetOffset(src_id);
          int64_t adj_offset =
              __sync_fetch_and_add(&offsets[v_label][v_offset], 1);
          nbr_unit_t* ptr = edges[v_label]->MutablePointer(adj_offset);
          ptr->vid = dst_list_ptr[i];
          ptr->eid = static_cast<EID_T>(chunk_offsets[chunk_index] + i);
        }
        src_chunks[chunk_index].reset();
        dst_chunks[chunk_index].reset();
      },
      concurrency);

  VLOG(100) << "Finish building the CSR ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    sort_edges_with_respect_to_vertex(*edges[v_label], edge_offsets[v_label],
                                      tvnums[v_label], concurrency);
    if (!is_multigraph) {
      check_is_multigraph(*edges[v_label], edge_offsets[v_label],
                          tvnums[v_label], concurrency, is_multigraph);
    }
  }
  return {};
}

template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_directed_csc(
    Client& client, IdParser<VID_T>& parser, std::vector<VID_T> tvnums,
    int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& oedges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& oedge_offsets,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& iedges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& iedge_offsets,
    bool& is_multigraph) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  std::vector<std::vector<int>> degree(vertex_label_num);
  std::vector<int64_t> actual_edge_num(vertex_label_num, 0);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    degree[v_label].resize(tvnums[v_label], 0);
  }

  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    const nbr_unit_t* oe = oedges[v_label]->MutablePointer(0);
    const int64_t* oe_offsets = oedge_offsets[v_label]->raw_values();
    parallel_for(
        static_cast<VID_T>(0), tvnums[v_label],
        [&degree, &parser, &v_label, &oe, &oe_offsets](VID_T src_offset) {
          for (int64_t i = oe_offsets[src_offset];
               i < oe_offsets[src_offset + 1]; ++i) {
            VID_T dst_id = oe[i].vid;
            grape::atomic_add(
                degree[parser.GetLabelId(dst_id)][parser.GetOffset(dst_id)], 1);
          }
        },
        concurrency);
  }

  std::vector<std::vector<int64_t>> offsets(vertex_label_num);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    auto tvnum = tvnums[v_label];
    auto& offset_vec = offsets[v_label];
    auto& degree_vec = degree[v_label];

    offset_vec.resize(tvnum + 1);
    offset_vec[0] = 0;

    if (tvnum > 0) {
      parallel_prefix_sum(degree_vec.data(), &offset_vec[1], tvnum,
                          concurrency);
    }
    // build the arrow's offset array
    std::shared_ptr<arrow::Buffer> offsets_buffer;
    ARROW_OK_ASSIGN_OR_RAISE(
        offsets_buffer, arrow::AllocateBuffer((tvnum + 1) * sizeof(int64_t),
                                              arrow::default_memory_pool()));
    memcpy(offsets_buffer->mutable_data(), offset_vec.data(),
           (tvnum + 1) * sizeof(int64_t));
    iedge_offsets[v_label] = std::make_shared<arrow::Int64Array>(
        arrow::int64(), tvnum + 1, offsets_buffer, nullptr, 0, 0);
    actual_edge_num[v_label] = offset_vec[tvnum];
  }
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    iedges[v_label] = std::make_shared<PodArrayBuilder<nbr_unit_t>>(
        client, actual_edge_num[v_label]);
  }

  VLOG(100) << "Start building the CSC ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    const nbr_unit_t* oe = oedges[v_label]->MutablePointer(0);
    const int64_t* oe_offsets = oedge_offsets[v_label]->raw_values();
    parallel_for(
        static_cast<VID_T>(0), tvnums[v_label],
        [&degree, &parser, &v_label, &offsets, &iedges, &oe,
         &oe_offsets](VID_T src_offset) {
          VID_T src_id = parser.GenerateId(v_label, src_offset);
          for (int64_t i = oe_offsets[src_offset];
               i < oe_offsets[src_offset + 1]; ++i) {
            VID_T dst_id = oe[i].vid;
            int u_label = parser.GetLabelId(dst_id);
            int64_t u_offset = parser.GetOffset(dst_id);
            int64_t adj_offset =
                __sync_fetch_and_add(&offsets[u_label][u_offset], 1);
            nbr_unit_t* ptr = iedges[u_label]->MutablePointer(adj_offset);
            ptr->vid = src_id;
            ptr->eid = static_cast<EID_T>(oe[i].eid);
          }
        },
        concurrency);
  }

  VLOG(100) << "Finish building the CSC ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    sort_edges_with_respect_to_vertex(*iedges[v_label], iedge_offsets[v_label],
                                      tvnums[v_label], concurrency);
    if (!is_multigraph) {
      check_is_multigraph(*iedges[v_label], iedge_offsets[v_label],
                          tvnums[v_label], concurrency, is_multigraph);
    }
  }
  return {};
}

template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_undirected_csr(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    bool& is_multigraph) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  int64_t num_chunks = src_chunks.size();
  std::vector<std::vector<int>> degree(vertex_label_num);
  std::vector<int64_t> actual_edge_num(vertex_label_num, 0);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    degree[v_label].resize(tvnums[v_label], 0);
  }

  // compute the degrees
  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&degree, &parser, &src_chunks, &dst_chunks](int64_t chunk_index) {
        auto src_array = src_chunks[chunk_index];
        auto dst_array = dst_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();
        const VID_T* dst_list_ptr = dst_array->raw_values();

        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          VID_T dst_id = dst_list_ptr[i];
          grape::atomic_add(
              degree[parser.GetLabelId(src_id)][parser.GetOffset(src_id)], 1);
          grape::atomic_add(
              degree[parser.GetLabelId(dst_id)][parser.GetOffset(dst_id)], 1);
        }
      },
      concurrency);

  // building the offsets array
  std::vector<std::vector<int64_t>> offsets(vertex_label_num);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    auto tvnum = tvnums[v_label];
    auto& offset_vec = offsets[v_label];
    auto& degree_vec = degree[v_label];

    offset_vec.resize(tvnum + 1);
    offset_vec[0] = 0;

    if (tvnum > 0) {
      parallel_prefix_sum(degree_vec.data(), &offset_vec[1], tvnum,
                          concurrency);
    }
    // build the arrow's offset array
    std::shared_ptr<arrow::Buffer> offsets_buffer;
    ARROW_OK_ASSIGN_OR_RAISE(
        offsets_buffer, arrow::AllocateBuffer((tvnum + 1) * sizeof(int64_t),
                                              arrow::default_memory_pool()));
    memcpy(offsets_buffer->mutable_data(), offset_vec.data(),
           (tvnum + 1) * sizeof(int64_t));
    edge_offsets[v_label] = std::make_shared<arrow::Int64Array>(
        arrow::int64(), tvnum + 1, offsets_buffer, nullptr, 0, 0);
    actual_edge_num[v_label] = offset_vec[tvnum];
  }

  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    edges[v_label] = std::make_shared<PodArrayBuilder<nbr_unit_t>>(
        client, actual_edge_num[v_label]);
  }

  VLOG(100) << "Start building the CSR ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);
  for (int64_t i = 0; i < num_chunks; ++i) {
    chunk_offsets[i + 1] = chunk_offsets[i] + src_chunks[i]->length();
  }
  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&src_chunks, &dst_chunks, &parser, &edges, &offsets,
       &chunk_offsets](int64_t chunk_index) {
        auto& src_array = src_chunks[chunk_index];
        auto& dst_array = dst_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();
        const VID_T* dst_list_ptr = dst_array->raw_values();
        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          VID_T dst_id = dst_list_ptr[i];
          auto src_label = parser.GetLabelId(src_id);
          int64_t src_offset = parser.GetOffset(src_id);
          auto dst_label = parser.GetLabelId(dst_id);
          int64_t dst_offset = parser.GetOffset(dst_id);

          int64_t oe_offset =
              __sync_fetch_and_add(&offsets[src_label][src_offset], 1);
          nbr_unit_t* src_ptr = edges[src_label]->MutablePointer(oe_offset);
          src_ptr->vid = dst_id;
          src_ptr->eid = static_cast<EID_T>(chunk_offsets[chunk_index] + i);

          int64_t ie_offset =
              __sync_fetch_and_add(&offsets[dst_label][dst_offset], 1);
          nbr_unit_t* dst_ptr = edges[dst_label]->MutablePointer(ie_offset);
          dst_ptr->vid = src_id;
          dst_ptr->eid = static_cast<EID_T>(chunk_offsets[chunk_index] + i);
        }
        src_chunks[chunk_index].reset();
        dst_chunks[chunk_index].reset();
      },
      concurrency);

  VLOG(100) << "Finish building the CSR ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    sort_edges_with_respect_to_vertex(*edges[v_label], edge_offsets[v_label],
                                      tvnums[v_label], concurrency);
    if (!is_multigraph) {
      check_is_multigraph(*edges[v_label], edge_offsets[v_label],
                          tvnums[v_label], concurrency, is_multigraph);
    }
  }
  return {};
}

template <typename VID_T, typename EID_T>
boost::leaf::result<void> generate_undirected_csr_memopt(
    Client& client, IdParser<VID_T>& parser,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> src_chunks,
    std::vector<std::shared_ptr<ArrowArrayType<VID_T>>> dst_chunks,
    std::vector<VID_T> tvnums, int vertex_label_num, int concurrency,
    std::vector<std::shared_ptr<
        PodArrayBuilder<property_graph_utils::NbrUnit<VID_T, EID_T>>>>& edges,
    std::vector<std::shared_ptr<arrow::Int64Array>>& edge_offsets,
    bool& is_multigraph) {
  using nbr_unit_t = property_graph_utils::NbrUnit<VID_T, EID_T>;

  int64_t num_chunks = src_chunks.size();
  std::vector<std::vector<int>> degree(vertex_label_num);
  std::vector<int64_t> actual_edge_num(vertex_label_num, 0);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    degree[v_label].resize(tvnums[v_label], 0);
  }

  // compute the degrees
  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&degree, &parser, &src_chunks, &dst_chunks](int64_t chunk_index) {
        auto src_array = src_chunks[chunk_index];
        auto dst_array = dst_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();
        const VID_T* dst_list_ptr = dst_array->raw_values();

        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          VID_T dst_id = dst_list_ptr[i];
          grape::atomic_add(
              degree[parser.GetLabelId(src_id)][parser.GetOffset(src_id)], 1);
          grape::atomic_add(
              degree[parser.GetLabelId(dst_id)][parser.GetOffset(dst_id)], 1);
        }
      },
      concurrency);

  // building the offsets array
  std::vector<std::vector<int64_t>> offsets(vertex_label_num);
  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    auto tvnum = tvnums[v_label];
    auto& offset_vec = offsets[v_label];
    auto& degree_vec = degree[v_label];

    offset_vec.resize(tvnum + 1);
    offset_vec[0] = 0;

    if (tvnum > 0) {
      parallel_prefix_sum(degree_vec.data(), &offset_vec[1], tvnum,
                          concurrency);
    }
    // build the arrow's offset array
    std::shared_ptr<arrow::Buffer> offsets_buffer;
    ARROW_OK_ASSIGN_OR_RAISE(
        offsets_buffer, arrow::AllocateBuffer((tvnum + 1) * sizeof(int64_t),
                                              arrow::default_memory_pool()));
    memcpy(offsets_buffer->mutable_data(), offset_vec.data(),
           (tvnum + 1) * sizeof(int64_t));
    edge_offsets[v_label] = std::make_shared<arrow::Int64Array>(
        arrow::int64(), tvnum + 1, offsets_buffer, nullptr, 0, 0);
    actual_edge_num[v_label] = offset_vec[tvnum];
  }

  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    edges[v_label] = std::make_shared<PodArrayBuilder<nbr_unit_t>>(
        client, actual_edge_num[v_label]);
  }

  VLOG(100) << "Start building the CSR ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);
  for (int64_t i = 0; i < num_chunks; ++i) {
    chunk_offsets[i + 1] = chunk_offsets[i] + src_chunks[i]->length();
  }
  parallel_for(
      static_cast<int64_t>(0), num_chunks,
      [&src_chunks, &dst_chunks, &parser, &edges, &offsets,
       &chunk_offsets](int64_t chunk_index) {
        auto& src_array = src_chunks[chunk_index];
        auto& dst_array = dst_chunks[chunk_index];
        const VID_T* src_list_ptr = src_array->raw_values();
        const VID_T* dst_list_ptr = dst_array->raw_values();
        for (int64_t i = 0; i < src_array->length(); ++i) {
          VID_T src_id = src_list_ptr[i];
          VID_T dst_id = dst_list_ptr[i];
          auto src_label = parser.GetLabelId(src_id);
          int64_t src_offset = parser.GetOffset(src_id);

          int64_t oe_offset =
              __sync_fetch_and_add(&offsets[src_label][src_offset], 1);
          nbr_unit_t* src_ptr = edges[src_label]->MutablePointer(oe_offset);
          src_ptr->vid = dst_id;
          src_ptr->eid = static_cast<EID_T>(chunk_offsets[chunk_index] + i);
        }
        src_chunks[chunk_index].reset();
        dst_chunks[chunk_index].reset();
      },
      concurrency);

  VLOG(100) << "Finish building the CSR ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  std::vector<std::vector<int64_t>> csr_offsets = offsets;  // make a copy

  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    const nbr_unit_t* oe = edges[v_label]->MutablePointer(0);
    const int64_t* oe_offsets = edge_offsets[v_label]->raw_values();
    parallel_for(
        static_cast<VID_T>(0), tvnums[v_label],
        [&degree, &parser, &v_label, &csr_offsets, &offsets, &oe_offsets,
         &edges, &oe](VID_T src_offset) {
          VID_T src_id = parser.GenerateId(v_label, src_offset);
          for (int64_t i = oe_offsets[src_offset];
               i < csr_offsets[v_label][src_offset]; ++i) {
            VID_T dst_id = oe[i].vid;
            int u_label = parser.GetLabelId(dst_id);
            int64_t u_offset = parser.GetOffset(dst_id);
            int64_t adj_offset =
                __sync_fetch_and_add(&offsets[u_label][u_offset], 1);
            nbr_unit_t* ptr = edges[u_label]->MutablePointer(adj_offset);
            ptr->vid = src_id;
            ptr->eid = static_cast<EID_T>(oe[i].eid);
          }
        },
        concurrency);
  }

  VLOG(100) << "Finish building the CSC ..." << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  for (int v_label = 0; v_label != vertex_label_num; ++v_label) {
    sort_edges_with_respect_to_vertex(*edges[v_label], edge_offsets[v_label],
                                      tvnums[v_label], concurrency);
    if (!is_multigraph) {
      check_is_multigraph(*edges[v_label], edge_offsets[v_label],
                          tvnums[v_label], concurrency, is_multigraph);
    }
  }
  return {};
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_PROPERTY_GRAPH_UTILS_IMPL_H_
