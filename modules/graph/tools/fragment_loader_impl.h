/**
 * Copyright 2020-2023 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODULES_GRAPH_TOOLS_FRAGMENT_LOADER_IMPL_H_
#define MODULES_GRAPH_TOOLS_FRAGMENT_LOADER_IMPL_H_

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "client/client.h"
#include "common/util/env.h"
#include "common/util/functions.h"
#include "common/util/json.h"
#include "common/util/logging.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/loader/arrow_fragment_loader.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/tools/graph_loader.h"

namespace vineyard {

namespace detail {

template <typename OID_T, typename VID_T>
ObjectID load_graph(Client& client, grape::CommSpec& comm_spec,
                    struct detail::loader_options const& options) {
  using loader_t = ArrowFragmentLoader<OID_T, VID_T>;
  auto loader = loader_t(client, comm_spec, std::vector<std::string>{},
                         std::vector<std::string>{}, options.directed,
                         options.generate_eid, options.retain_oid,
                         options.local_vertex_map, options.compact_edges,
                         options.use_perfect_hash);
  MPI_Barrier(comm_spec.comm());
  auto fn = [&]() -> boost::leaf::result<ObjectID> {
    return loader.LoadFragmentAsFragmentGroup(options.efiles, options.vfiles);
  };
  auto wholefn = [&]() -> boost::leaf::result<ObjectID> {
    ObjectID frag = InvalidObjectID();
    BOOST_LEAF_ASSIGN(frag, loader.LoadFragment());
    loader_t adder(client, comm_spec, options.efiles, options.vfiles,
                   options.directed, options.generate_eid, options.retain_oid);
    return adder.AddLabelsToFragmentAsFragmentGroup(frag);
  };
  auto stepbystepfn = [&]() -> boost::leaf::result<ObjectID> {
    ObjectID frag = InvalidObjectID();
    BOOST_LEAF_ASSIGN(frag, loader.LoadFragment());
    int64_t vlabel_num = 0;
    for (auto vfile : options.vfiles) {
      std::vector<std::string> sub_files;
      boost::split(sub_files, vfile, boost::is_any_of(";"));
      for (size_t i = 0; i < sub_files.size(); ++i) {
        loader_t adder(client, comm_spec, std::vector<std::string>{},
                       std::vector<std::string>{sub_files[i]}, options.directed,
                       options.generate_eid, options.retain_oid);
        if (i == 0) {
          BOOST_LEAF_ASSIGN(frag, adder.AddLabelsToFragment(frag));
        } else {
          BOOST_LEAF_ASSIGN(frag,
                            adder.AddDataToExistedVLabel(frag, vlabel_num));
        }
      }
      vlabel_num++;
    }
    int64_t elabel_num = 0;
    for (auto efile : options.efiles) {
      std::vector<std::string> sub_files;
      boost::split(sub_files, efile, boost::is_any_of(";"));
      for (size_t i = 0; i < sub_files.size(); ++i) {
        loader_t adder(client, comm_spec,
                       std::vector<std::string>{sub_files[i]},
                       std::vector<std::string>{}, options.directed,
                       options.generate_eid, options.retain_oid);
        if (i == 0) {
          BOOST_LEAF_ASSIGN(frag, adder.AddLabelsToFragment(frag));
        } else {
          BOOST_LEAF_ASSIGN(frag,
                            adder.AddDataToExistedELabel(frag, elabel_num));
        }
      }
      elabel_num++;
    }
    return vineyard::ConstructFragmentGroup(client, frag, comm_spec);
  };

  auto loadfn = [&]() -> boost::leaf::result<ObjectID> {
    if (options.progressive == progressive_t::WHOLE) {
      return wholefn();
    }
    if (options.progressive == progressive_t::STEP_BY_STEP) {
      return stepbystepfn();
    }
    if (true /* options.progressive == progressive_t::NONE */) {
      return fn();
    }
  };

  if (options.catch_leaf_errors) {
    return boost::leaf::try_handle_all(
        [&loadfn]() { return loadfn(); },
        [](const GSError& e) {
          LOG(ERROR) << e.error_msg;
          return InvalidObjectID();
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(ERROR) << "Unmatched error " << unmatched;
          return InvalidObjectID();
        });
  } else {
    return loadfn().value();
  }
}

template <typename fragment_t>
void traverse_fragment(const std::shared_ptr<fragment_t>& fragment,
                       const size_t rounds = 1) {
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  label_id_t e_label_num = fragment->edge_label_num();
  label_id_t v_label_num = fragment->vertex_label_num();

  LOG(INFO) << "start traversing the fragment ...";
  auto start_time = std::chrono::high_resolution_clock::now();

  typename fragment_t::vertex_t neighbor;
  [[maybe_unused]] volatile typename fragment_t::eid_t neighbor_edge;
  for (size_t round = 0; round < rounds; ++round) {
    for (label_id_t v_label = 0; v_label != v_label_num; ++v_label) {
      auto iv = fragment->InnerVertices(v_label);
      for (auto v : iv) {
        for (label_id_t e_label = 0; e_label != e_label_num; ++e_label) {
          auto oe = fragment->GetOutgoingAdjList(v, e_label);
          for (auto& e : oe) {
            neighbor = e.neighbor();
            neighbor_edge = e.edge_id();
          }
          if (fragment->directed()) {
            auto ie = fragment->GetIncomingAdjList(v, e_label);
            for (auto& e : ie) {
              neighbor = e.neighbor();
              neighbor_edge = e.edge_id();
            }
          }
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "traversing the fragment for " << rounds << " rounds takes "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   end_time - start_time)
                   .count()
            << " seconds";
}

template <typename fragment_t>
void dump_fragment(const std::shared_ptr<fragment_t>& fragment,
                   const std::string& path_prefix) {
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  label_id_t e_label_num = fragment->edge_label_num();
  label_id_t v_label_num = fragment->vertex_label_num();

  LOG(INFO) << "start dumping the fragment ...";
  auto start_time = std::chrono::high_resolution_clock::now();

  for (label_id_t v_label = 0; v_label != v_label_num; ++v_label) {
    std::ofstream fout(path_prefix + "_v" + std::to_string(v_label),
                       std::ios::binary);
    auto iv = fragment->InnerVertices(v_label);
    for (auto v : iv) {
      auto id = fragment->GetId(v);
      fout << id << std::endl;
    }
    fout.flush();
    fout.close();
  }
  for (label_id_t e_label = 0; e_label != e_label_num; ++e_label) {
    std::ofstream fout(path_prefix + "_e" + std::to_string(e_label),
                       std::ios::binary);
    for (label_id_t v_label = 0; v_label != v_label_num; ++v_label) {
      auto iv = fragment->InnerVertices(v_label);
      for (auto v : iv) {
        auto src_id = fragment->GetId(v);
        auto oe = fragment->GetOutgoingAdjList(v, e_label);
        for (auto& e : oe) {
          fout << src_id << " " << fragment->GetId(e.neighbor()) << "\n";
        }
        if (fragment->directed()) {
          auto ie = fragment->GetIncomingAdjList(v, e_label);
          for (auto& e : ie) {
            fout << fragment->GetId(e.neighbor()) << " " << src_id << "\n";
          }
        }
      }
    }
    fout.flush();
    fout.close();
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "dumping the fragment takes "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   end_time - start_time)
                   .count()
            << " seconds";
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T,
          bool COMPACT = false>
void dump_graph_impl(Client& client, grape::CommSpec& comm_spec,
                     const ObjectID fragment_group_id,
                     struct detail::loader_options const& options) {
  if (comm_spec.local_id() != 0) {
    return;
  }
  const std::string target_directory = options.dump;
  create_dirs(target_directory.c_str());

  using fragment_t =
      ArrowFragment<OID_T, VID_T,
                    VERTEX_MAP_T<typename InternalType<OID_T>::type, VID_T>,
                    COMPACT>;

  std::shared_ptr<vineyard::ArrowFragmentGroup> fg =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(
          client.GetObject(fragment_group_id));

  auto const& fragments = fg->Fragments();
  for (const auto& item : fg->FragmentLocations()) {
    if (item.second == client.instance_id()) {
      std::shared_ptr<fragment_t> fragment;
      VINEYARD_CHECK_OK(client.GetObject(fragments.at(item.first), fragment));
      if (options.dump_dry_run_rounds) {
        traverse_fragment(fragment, options.dump_dry_run_rounds);
      } else {
        dump_fragment(fragment, target_directory + "/output_graph_f" +
                                    std::to_string(fragment->fid()));
      }
    }
  }
}

template <typename OID_T, typename VID_T>
void dump_graph(Client& client, grape::CommSpec& comm_spec,
                const ObjectID fragment_group_id,
                struct detail::loader_options const& options) {
  if (!options.local_vertex_map && !options.compact_edges) {
    dump_graph_impl<OID_T, VID_T, ArrowVertexMap, false>(
        client, comm_spec, fragment_group_id, options);
  } else if (!options.local_vertex_map && options.compact_edges) {
    dump_graph_impl<OID_T, VID_T, ArrowVertexMap, true>(
        client, comm_spec, fragment_group_id, options);
  } else if (options.local_vertex_map && !options.compact_edges) {
    dump_graph_impl<OID_T, VID_T, ArrowLocalVertexMap, false>(
        client, comm_spec, fragment_group_id, options);
  } else {
    dump_graph_impl<OID_T, VID_T, ArrowLocalVertexMap, true>(
        client, comm_spec, fragment_group_id, options);
  }
}

}  // namespace detail

}  // namespace vineyard

#endif  // MODULES_GRAPH_TOOLS_FRAGMENT_LOADER_IMPL_H_
