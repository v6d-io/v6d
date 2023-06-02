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

#ifndef MODULES_GRAPH_FRAGMENT_FRAGMENT_TRAITS_H_
#define MODULES_GRAPH_FRAGMENT_FRAGMENT_TRAITS_H_

#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "common/util/typename.h"

namespace vineyard {

// TODO(lxj): unify the types conversion or place them together? e.g.
// those in context_utils.h
template <typename T>
struct is_property_fragment {
  using type = std::false_type;
  static constexpr bool value = false;
};

class ArrowFragmentBase;

template <>
struct is_property_fragment<ArrowFragmentBase> {
  using type = std::true_type;
  static constexpr bool value = true;
};

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
class ArrowFragment;

// Currently `typename_t` cannot automatically process non-type template
// parameters
template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
struct typename_t<ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>> {
  inline static const std::string name() {
    std::ostringstream ss;
    ss << "vineyard::ArrowFragment<";
    ss << type_name<OID_T>() << ",";
    ss << type_name<VID_T>() << ",";
    ss << type_name<VERTEX_MAP_T>() << ",";
    ss << (COMPACT ? "true" : "false") << ">";
    return ss.str();
  }
};

template <typename OID_T, typename VID_T, typename VERTEX_MAP_T, bool COMPACT>
struct is_property_fragment<
    ArrowFragment<OID_T, VID_T, VERTEX_MAP_T, COMPACT>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

}  // namespace vineyard

namespace gs {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T, bool COMPACT>
class ArrowProjectedFragment;

}  // namespace gs

namespace vineyard {

// Currently `typename_t` cannot automatically process non-type template
// parameters
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T, bool COMPACT>
struct typename_t<gs::ArrowProjectedFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                             VERTEX_MAP_T, COMPACT>> {
  inline static const std::string name() {
    std::ostringstream ss;
    ss << "gs::ArrowProjectedFragment<";
    ss << type_name<OID_T>() << ",";
    ss << type_name<VID_T>() << ",";
    ss << type_name<VDATA_T>() << ",";
    ss << type_name<EDATA_T>() << ",";
    ss << type_name<VERTEX_MAP_T>() << ",";
    ss << (COMPACT ? "true" : "false") << ">";
    return ss.str();
  }
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_FRAGMENT_TRAITS_H_
