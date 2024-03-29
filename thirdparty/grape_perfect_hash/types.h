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

#ifndef GRAPE_TYPES_H_
#define GRAPE_TYPES_H_

#include <istream>
#include <ostream>
#include <type_traits>
#include "pthash/utils/hasher.hpp"

// Use the same setting with apache-arrow to avoid possible conflicts
#define nssv_CONFIG_SELECT_STRING_VIEW nssv_STRING_VIEW_NONSTD
#include "string_view/string_view.hpp"

namespace grape {

/**
 * @brief EmptyType is the placeholder of VDATA_T and EDATA_T for graphs without
 * data on vertices and edges.
 *
 */
struct EmptyType {};

inline std::ostream& operator<<(std::ostream& out, const EmptyType) {
  return out;
}
inline std::istream& operator>>(std::istream& in, EmptyType) { return in; }

template <typename E>
using enable_enum_t =
    typename std::enable_if<std::is_enum<E>::value,
                            typename std::underlying_type<E>::type>::type;

template <typename E>
constexpr inline enable_enum_t<E> underlying_value(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

template <typename E, typename T>
constexpr inline typename std::enable_if<
    std::is_enum<E>::value && std::is_integral<T>::value, E>::type
to_enum(T value) noexcept {
  return static_cast<E>(value);
}

/**
 * @brief LoadStrategy specifies the which edges should be loadded when building
 * the graph from a location.
 *
 */
enum class LoadStrategy {
  kOnlyOut = 0,
  kOnlyIn = 1,
  kBothOutIn = 2,
  kNullLoadStrategy = 0xf0,
};

/**
 * @brief MessageStrategy specifies the method of message passing between
 * fragments.
 *
 * Assume in an edgecut distributed graph, we have an edge a->b, with vertex a
 * on fragment_1 and b on fragment_2, and an edge a<-c with c on f_2.
 *
 * for fragment_1, a is an inner_vertex and b', c' is outer_vertexs.
 *
 */
enum class MessageStrategy {
  kAlongOutgoingEdgeToOuterVertex = 0,  /// from a to b;
  kAlongIncomingEdgeToOuterVertex = 1,  /// from c to a;
  kAlongEdgeToOuterVertex = 2,          /// from a to b, a to c;
  kSyncOnOuterVertex = 3,               /// from b' to b and c' to c;
};

template <typename APP_T, typename GRAPH_T>
constexpr inline bool check_load_strategy_compatible() {
  return ((APP_T::load_strategy == LoadStrategy::kBothOutIn) &&
          (GRAPH_T::load_strategy == LoadStrategy::kBothOutIn)) ||
         ((APP_T::load_strategy == LoadStrategy::kOnlyIn) &&
          ((GRAPH_T::load_strategy == LoadStrategy::kBothOutIn) ||
           (GRAPH_T::load_strategy == LoadStrategy::kOnlyIn))) ||
         ((APP_T::load_strategy == LoadStrategy::kOnlyOut) &&
          ((GRAPH_T::load_strategy == LoadStrategy::kBothOutIn) ||
           (GRAPH_T::load_strategy == LoadStrategy::kOnlyOut)));
}

template <typename APP_T, typename GRAPH_T>
constexpr inline bool check_message_strategy_valid() {
  return ((APP_T::message_strategy ==
           MessageStrategy::kAlongEdgeToOuterVertex) &&
          (GRAPH_T::load_strategy == LoadStrategy::kBothOutIn)) ||
         ((APP_T::message_strategy ==
           MessageStrategy::kAlongIncomingEdgeToOuterVertex) &&
          ((GRAPH_T::load_strategy == LoadStrategy::kOnlyIn) ||
           (GRAPH_T::load_strategy == LoadStrategy::kBothOutIn))) ||
         ((APP_T::message_strategy ==
           MessageStrategy::kAlongOutgoingEdgeToOuterVertex) &&
          ((GRAPH_T::load_strategy == LoadStrategy::kOnlyOut) ||
           (GRAPH_T::load_strategy == LoadStrategy::kBothOutIn))) ||
         (APP_T::message_strategy == MessageStrategy::kSyncOnOuterVertex);
}

template <typename APP_T, typename GRAPH_T>
constexpr inline bool check_app_fragment_consistency() {
  return check_load_strategy_compatible<APP_T, GRAPH_T>() &&
         check_message_strategy_valid<APP_T, GRAPH_T>();
}

template <typename T>
struct InternalOID {
  using type = T;

  static type ToInternal(const T& val) { return val; }

  static T FromInternal(const type& val) { return val; }
};

template <>
struct InternalOID<std::string> {
  using type = nonstd::string_view;

  static type ToInternal(const std::string& val) {
    return nonstd::string_view(val.data(), val.size());
  }

  static std::string FromInternal(const type& val) { return std::string(val); }
};

struct murmurhasher {
  typedef pthash::hash64 hash_type;

  // specialization for std::string
  static inline hash_type hash(std::string const& val, uint64_t seed) {
    return pthash::MurmurHash2_64(val.data(), val.size(), seed);
  }

  // specialization for uint64_t
  static inline hash_type hash(uint64_t val, uint64_t seed) {
    return pthash::MurmurHash2_64(reinterpret_cast<char const*>(&val),
                                  sizeof(val), seed);
  }

  static inline hash_type hash(const nonstd::string_view& val, uint64_t seed) {
    return pthash::MurmurHash2_64(val.data(), val.size(), seed);
  }
};

}  // namespace grape

#endif  // GRAPE_TYPES_H_
