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

#ifndef MODULES_GRAPH_UTILS_SELECTOR_UTILS_H_
#define MODULES_GRAPH_UTILS_SELECTOR_UTILS_H_

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "grape/communication/sync_comm.h"
#include "grape/serialization/in_archive.h"
#include "grape/worker/comm_spec.h"

#include "graph/fragment/property_graph_types.h"

namespace vineyard {

enum class SchemaType {
  kVertexDataSchema,      // id/data/result of all vertices
  kVertexPropertySchema,  // id/data/result of single type(label) vertices
  kEdgeDataSchema,        // endpoint-id/data/(result) of all edges
  kEdgePropertySchema,  // endpoint-id/data/(result) of single type(label) edges
  kInvalidSchema,
};

inline bool parse_label_id(const std::string& token,
                           property_graph_types::LABEL_ID_TYPE& label_id) {
  if (token.length() <= 5) {
    return false;
  }
  std::string prefix = token.substr(0, 5);
  std::string suffix = token.substr(5);
  if (prefix == "label") {
    label_id = boost::lexical_cast<property_graph_types::LABEL_ID_TYPE>(suffix);
    return true;
  }
  return false;
}

inline bool parse_property_id(const std::string& token,
                              property_graph_types::PROP_ID_TYPE& prop_id) {
  if (token.length() <= 8) {
    return false;
  }
  std::string prefix = token.substr(0, 8);
  std::string suffix = token.substr(8);
  if (prefix == "property") {
    prop_id = boost::lexical_cast<property_graph_types::PROP_ID_TYPE>(suffix);
    return true;
  }
  return false;
}

inline std::string generate_selectors(
    const std::vector<std::pair<std::string, std::string>>& selector_list) {
  boost::property_tree::ptree tree;
  for (auto& pair : selector_list) {
    tree.put(pair.first, pair.second);
  }
  std::stringstream ss;
  boost::property_tree::json_parser::write_json(ss, tree, false);
  return ss.str();
}

inline SchemaType parse_selectors(
    const std::string& selectors,
    std::vector<std::pair<std::string, std::string>>& selector_list) {
  std::stringstream ss(selectors);
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::json_parser::read_json(ss, pt);
    BOOST_FOREACH  // NOLINT(whitespace/parens)
        (boost::property_tree::ptree::value_type & v, pt) {
      CHECK(v.second.empty());
      selector_list.emplace_back(v.first, v.second.data());
    }
  } catch (boost::property_tree::ptree_error& e) {
    return SchemaType::kInvalidSchema;
  }

  bool schema_for_vertex = false;
  bool schema_for_edge = false;
  bool schema_for_context = false;

  for (auto& pair : selector_list) {
    std::vector<std::string> token;
    boost::split(token, pair.second, boost::is_any_of("."));
    if (token[0] == "v") {
      schema_for_vertex = true;
      if (token.size() != 2) {
        return SchemaType::kInvalidSchema;
      }
      if (token[1] != "data" && token[1] != "id") {
        return SchemaType::kInvalidSchema;
      }
    } else if (token[0] == "e") {
      schema_for_edge = true;
      if (token.size() != 2) {
        return SchemaType::kInvalidSchema;
      }
      if (token[1] != "data" && token[1] != "src" && token[1] != "dst") {
        return SchemaType::kInvalidSchema;
      }
    } else if (token[0] == "r") {
      schema_for_context = true;
      if (token.size() != 1) {
        return SchemaType::kInvalidSchema;
      }
    }
  }

  if ((schema_for_vertex || schema_for_context) && !schema_for_edge) {
    return SchemaType::kVertexDataSchema;
  }
  if (!schema_for_vertex && !schema_for_context && schema_for_edge) {
    return SchemaType::kEdgeDataSchema;
  }
  return SchemaType::kInvalidSchema;
}

inline SchemaType parse_property_selectors(
    const std::string& selectors,
    std::vector<std::pair<std::string, std::string>>& selector_list,
    property_graph_types::LABEL_ID_TYPE& label_id_out) {
  std::stringstream ss(selectors);
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::read_json(ss, pt);
    BOOST_FOREACH  // NOLINT(whitespace/parens)
        (boost::property_tree::ptree::value_type & v, pt) {
      CHECK(v.second.empty());
      selector_list.emplace_back(v.first, v.second.data());
    }
  } catch (boost::property_tree::ptree_error& e) {
    return SchemaType::kInvalidSchema;
  }

  bool schema_for_vertex = false;
  bool schema_for_edge = false;
  bool schema_for_context = false;

  std::set<property_graph_types::LABEL_ID_TYPE> v_labels, e_labels;

  for (auto& pair : selector_list) {
    std::vector<std::string> token;
    boost::split(token, pair.second, boost::is_any_of("."));
    if (token[0] == "v") {
      schema_for_vertex = true;
      if (token.size() != 3) {
        return SchemaType::kInvalidSchema;
      }
      property_graph_types::LABEL_ID_TYPE label_id;
      property_graph_types::PROP_ID_TYPE prop_id;

      if (parse_label_id(token[1], label_id)) {
        if (token[2] == "id" || parse_property_id(token[2], prop_id)) {
          v_labels.insert(label_id);
        } else {
          return SchemaType::kInvalidSchema;
        }
      } else {
        return SchemaType::kInvalidSchema;
      }
    } else if (token[0] == "e") {
      schema_for_edge = true;
      if (token.size() != 3) {
        return SchemaType::kInvalidSchema;
      }
      property_graph_types::LABEL_ID_TYPE label_id;
      property_graph_types::PROP_ID_TYPE prop_id;
      if (parse_label_id(token[1], label_id) &&
          (token[2] == "src" || token[2] == "dst" ||
           parse_property_id(token[2], prop_id))) {
        e_labels.insert(label_id);
      } else {
        return SchemaType::kInvalidSchema;
      }
    } else if (token[0] == "r") {
      schema_for_context = true;
      property_graph_types::LABEL_ID_TYPE label_id;
      property_graph_types::PROP_ID_TYPE prop_id;
      if (parse_label_id(token[1], label_id)) {
        if (token[2] == "data" || parse_property_id(token[2], prop_id)) {
          v_labels.insert(label_id);
        } else {
          return SchemaType::kInvalidSchema;
        }
      } else {
        return SchemaType::kInvalidSchema;
      }
    }
  }

  if ((schema_for_vertex || schema_for_context) && !schema_for_edge &&
      v_labels.size() == 1) {
    label_id_out = *v_labels.begin();
    return SchemaType::kVertexPropertySchema;
  }
  if (schema_for_edge && !schema_for_vertex && !schema_for_context &&
      e_labels.size() == 1) {
    label_id_out = *e_labels.begin();
    return SchemaType::kEdgePropertySchema;
  }
  return SchemaType::kInvalidSchema;
}

inline bool parse_add_column_selectors(
    const std::string& selectors,
    std::map<property_graph_types::LABEL_ID_TYPE,
             std::vector<std::pair<std::string, std::string>>>& selector_map) {
  std::stringstream ss(selectors);
  boost::property_tree::ptree pt;
  std::vector<std::pair<std::string, std::string>> tmp_list;
  try {
    boost::property_tree::read_json(ss, pt);
    BOOST_FOREACH  // NOLINT(whitespace/parens)
    (boost::property_tree::ptree::value_type & v, pt) {
            CHECK(v.second.empty());
            tmp_list.emplace_back(v.first, v.second.data());
          }
  } catch (boost::property_tree::ptree_error& e) { return false; }

  for (auto& pair : tmp_list) {
    std::vector<std::string> tokens;
    boost::split(tokens, pair.second, boost::is_any_of("."));
    if (tokens.size() >= 2) {
      if (tokens[0] == "v" || tokens[0] == "r") {
        property_graph_types::LABEL_ID_TYPE label_id;
        if (parse_label_id(tokens[1], label_id)) {
          selector_map[label_id].emplace_back(pair.first, pair.second);
        }
      }
    }
  }
  return true;
}

inline bool parse_no_labeled_add_column_selectors(
    const std::string& selectors,
    std::vector<std::pair<std::string, std::string>>& selector_map) {
  std::stringstream ss(selectors);
  boost::property_tree::ptree pt;
  std::vector<std::pair<std::string, std::string>> tmp_list;
  try {
    boost::property_tree::read_json(ss, pt);
    BOOST_FOREACH  // NOLINT(whitespace/parens)
        (boost::property_tree::ptree::value_type & v, pt) {
      CHECK(v.second.empty());
      const auto& col_name = v.first;
      const auto& selector = v.second.data();
      selector_map.emplace_back(col_name, selector);
    }
  } catch (boost::property_tree::ptree_error& e) { return false; }

  return true;
}

inline void gather_archives(grape::InArchive& arc,
                            const grape::CommSpec& comm_spec, size_t from = 0) {
  if (comm_spec.fid() == 0) {
    int64_t local_length = 0;
    std::vector<int64_t> gathered_length(comm_spec.fnum(), 0);
    MPI_Gather(&local_length, 1, MPI_INT64_T, &gathered_length[0], 1,
               MPI_INT64_T, comm_spec.worker_id(), comm_spec.comm());
    int64_t total_length = 0;
    for (auto gl : gathered_length) {
      total_length += gl;
    }
    size_t old_length = arc.GetSize();
    arc.Resize(old_length + total_length);
    char* ptr = arc.GetBuffer() + static_cast<ptrdiff_t>(old_length);

    for (grape::fid_t i = 1; i < comm_spec.fnum(); ++i) {
      grape::recv_buffer<char>(ptr, static_cast<size_t>(gathered_length[i]),
                               comm_spec.FragToWorker(i), comm_spec.comm(), 0);
      ptr += static_cast<ptrdiff_t>(gathered_length[i]);
    }
  } else {
    int64_t local_length = static_cast<int64_t>(arc.GetSize() - from);
    MPI_Gather(&local_length, 1, MPI_INT64_T, NULL, 1, MPI_INT64_T,
               comm_spec.FragToWorker(0), comm_spec.comm());

    grape::send_buffer<char>(arc.GetBuffer() + static_cast<ptrdiff_t>(from),
                             static_cast<size_t>(local_length),
                             comm_spec.FragToWorker(0), comm_spec.comm(), 0);
    arc.Resize(from);
  }
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_SELECTOR_UTILS_H_
