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

#ifndef MODULES_GRAPH_UTILS_TRANSFORM_UTILS_H_
#define MODULES_GRAPH_UTILS_TRANSFORM_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include <boost/lexical_cast.hpp>

namespace vineyard {

template <typename OID_T>
class String2Oid {
  using oid_t = OID_T;

 public:
  explicit String2Oid(std::string s_oid) : s_oid_(std::move(s_oid)) {}

  oid_t Value() { return boost::lexical_cast<oid_t>(s_oid_); }

 private:
  std::string s_oid_;
};

template <typename FRAG_T>
std::vector<typename FRAG_T::vertex_t> select_vertices(const FRAG_T* frag,
                                                       const std::string& begin,
                                                       const std::string& end) {
  using vertex_t = typename FRAG_T::vertex_t;
  using oid_t = typename FRAG_T::oid_t;
  std::vector<vertex_t> ret;
  auto range = frag->InnerVertices();
  if (begin == "" && end == "") {
    for (auto v : range) {
      ret.emplace_back(v);
    }
  } else if (begin == "" && end != "") {
    oid_t end_id = String2Oid<oid_t>(end).Value();
    for (auto v : range) {
      if (frag->GetId(v) < end_id) {
        ret.emplace_back(v);
      }
    }
  } else if (begin != "" && end == "") {
    oid_t begin_id = String2Oid<oid_t>(begin).Value();
    for (auto v : range) {
      if (frag->GetId(v) >= begin_id) {
        ret.emplace_back(v);
      }
    }
  } else if (begin != "" && end != "") {
    oid_t begin_id = String2Oid<oid_t>(begin).Value();
    oid_t end_id = String2Oid<oid_t>(end).Value();
    for (auto v : range) {
      oid_t id = frag->GetId(v);
      if (id >= begin_id && id < end_id) {
        ret.emplace_back(v);
      }
    }
  }
  return ret;
}

template <typename FRAG_T>
std::vector<typename FRAG_T::vertex_t> select_labeled_vertices(
    const FRAG_T* frag, typename FRAG_T::label_id_t label_id,
    const std::string& begin, const std::string& end) {
  using vertex_t = typename FRAG_T::vertex_t;
  using oid_t = typename FRAG_T::oid_t;

  std::vector<vertex_t> ret;
  auto range = frag->InnerVertices(label_id);
  if (begin == "" && end == "") {
    for (auto v : range) {
      ret.emplace_back(v);
    }
  } else if (begin == "" && end != "") {
    oid_t end_id = String2Oid<oid_t>(end).Value();
    for (auto v : range) {
      if (frag->GetId(v) < end_id) {
        ret.emplace_back(v);
      }
    }
  } else if (begin != "" && end == "") {
    oid_t begin_id = String2Oid<oid_t>(begin).Value();
    for (auto v : range) {
      if (frag->GetId(v) >= begin_id) {
        ret.emplace_back(v);
      }
    }
  } else if (begin != "" && end != "") {
    oid_t begin_id = String2Oid<oid_t>(begin).Value();
    oid_t end_id = String2Oid<oid_t>(end).Value();
    for (auto v : range) {
      oid_t id = frag->GetId(v);
      if (id >= begin_id && id < end_id) {
        ret.emplace_back(v);
      }
    }
  }
  return ret;
}

template <typename FRAG_T>
void serialize_vertex_id(grape::InArchive& arc, const FRAG_T* frag,
                         const std::vector<typename FRAG_T::vertex_t>& range) {
  for (auto v : range) {
    arc << frag->GetId(v);
  }
}

template <typename FRAG_T>
void serialize_vertex_data(
    grape::InArchive& arc, const FRAG_T* frag,
    const std::vector<typename FRAG_T::vertex_t>& range) {
  for (auto v : range) {
    arc << frag->GetData(v);
  }
}

template <typename FRAG_T, typename DATA_T>
void serialize_vertex_property_impl(
    grape::InArchive& arc, const FRAG_T* frag,
    const std::vector<typename FRAG_T::vertex_t>& range,
    typename FRAG_T::prop_id_t prop_id) {
  for (auto v : range) {
    arc << frag->template GetData<DATA_T>(v, prop_id);
  }
}

template <typename FRAG_T>
void serialize_vertex_property(
    grape::InArchive& arc, const FRAG_T* frag,
    const std::vector<typename FRAG_T::vertex_t>& range,
    typename FRAG_T::label_id_t label_id, typename FRAG_T::prop_id_t prop_id) {
  auto type = frag->vertex_property_type(label_id, prop_id);
  if (type->Equals(arrow::int32())) {
    serialize_vertex_property_impl<FRAG_T, int32_t>(arc, frag, range, prop_id);
  } else if (type->Equals(arrow::int64())) {
    serialize_vertex_property_impl<FRAG_T, int64_t>(arc, frag, range, prop_id);
  } else if (type->Equals(arrow::uint32())) {
    serialize_vertex_property_impl<FRAG_T, uint32_t>(arc, frag, range, prop_id);
  } else if (type->Equals(arrow::uint64())) {
    serialize_vertex_property_impl<FRAG_T, int64_t>(arc, frag, range, prop_id);
  } else if (type->Equals(arrow::float32())) {
    serialize_vertex_property_impl<FRAG_T, float>(arc, frag, range, prop_id);
  } else if (type->Equals(arrow::float64())) {
    serialize_vertex_property_impl<FRAG_T, double>(arc, frag, range, prop_id);
  } else {
    LOG(FATAL) << "property type not support - " << type->ToString();
  }
}

inline void parse_range(const std::string& range, std::string& begin,
                        std::string& end) {
  // format: "{begin: a, end: b}" or "{begin: a}" or "{end: b}" or "{}"
  std::stringstream ss(range);
  boost::property_tree::ptree pt;
  begin = "";
  end = "";
  try {
    boost::property_tree::json_parser::read_json(ss, pt);
    BOOST_FOREACH  // NOLINT(whitespace/parens)
        (boost::property_tree::ptree::value_type & v, pt) {
      CHECK(v.second.empty());
      if (v.first == "begin") {
        begin = v.second.data();
      }
      if (v.first == "end") {
        end = v.second.data();
      }
    }
  } catch (boost::property_tree::ptree_error& e) {
    begin = "";
    end = "";
  }
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TRANSFORM_UTILS_H_
