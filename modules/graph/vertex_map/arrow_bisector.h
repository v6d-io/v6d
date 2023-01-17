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

#ifndef MODULES_GRAPH_VERTEX_MAP_ARROW_BISECTOR_H_
#define MODULES_GRAPH_VERTEX_MAP_ARROW_BISECTOR_H_

#include <memory>

#include "basic/ds/arrow.h"

namespace vineyard {

// keep for backwards compatibility.
template <typename T>
class ArrowBisector {
 public:
  ArrowBisector();

  ArrowBisector(std::shared_ptr<ArrowArrayType<T>> const& values,
                std::shared_ptr<arrow::UInt64Array> const& sorted_indices,
                uint64_t offset = 0);

  ArrowBisector(const ArrowBisector& bisector);

  ArrowBisector(ArrowBisector&& bisector);

  ArrowBisector& operator=(const ArrowBisector& bisector);

  ArrowBisector& operator=(ArrowBisector&& bisector);

  inline int64_t find(T const& v) const;

  inline bool has(T const& v) const;

 private:
  std::shared_ptr<ArrowArrayType<T>> values_;
  std::shared_ptr<arrow::UInt64Array> sorted_indices_;

  int64_t length_ = 0;
  const uint64_t* indices_ = nullptr;
  uint64_t index_offset_ = 0;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_VERTEX_MAP_ARROW_BISECTOR_H_
