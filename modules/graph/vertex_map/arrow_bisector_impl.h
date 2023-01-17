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

#ifndef MODULES_GRAPH_VERTEX_MAP_ARROW_BISECTOR_IMPL_H_
#define MODULES_GRAPH_VERTEX_MAP_ARROW_BISECTOR_IMPL_H_

#include <memory>
#include <utility>

#include "basic/ds/arrow.h"
#include "basic/ds/arrow_utils.h"

#include "graph/vertex_map/arrow_bisector.h"

namespace vineyard {

// keep for backwards compatibility.
template <typename T>
ArrowBisector<T>::ArrowBisector()
    : values_(nullptr),
      sorted_indices_(nullptr),
      length_(0),
      indices_(nullptr),
      index_offset_(0) {}

template <typename T>
ArrowBisector<T>::ArrowBisector(
    std::shared_ptr<ArrowArrayType<T>> const& values,
    std::shared_ptr<arrow::UInt64Array> const& sorted_indices, uint64_t offset)
    : values_(values),
      sorted_indices_(sorted_indices),
      length_(values->length()),
      indices_(sorted_indices->raw_values()) {}

template <typename T>
ArrowBisector<T>::ArrowBisector(const ArrowBisector& bisector) {
  this->values_ = bisector.values_;
  this->sorted_indices_ = bisector.sorted_indices_;
  this->length_ = bisector.length_;
  this->indices_ = bisector.indices_;
  this->index_offset_ = bisector.index_offset_;
}

template <typename T>
ArrowBisector<T>::ArrowBisector(ArrowBisector&& bisector) {
  this->values_ = std::move(bisector.values_);
  this->sorted_indices_ = std::move(bisector.sorted_indices_);
  this->length_ = bisector.length_;
  this->indices_ = bisector.indices_;
  this->index_offset_ = bisector.index_offset_;

  bisector.length_ = 0;
  bisector.sorted_indices_ = nullptr;
}

template <typename T>
ArrowBisector<T>& ArrowBisector<T>::operator=(const ArrowBisector& bisector) {
  if (this != &bisector) {  // not a self-assignment
    this->values_ = bisector.values_;
    this->sorted_indices_ = bisector.sorted_indices_;
    this->length_ = bisector.length_;
    this->indices_ = bisector.indices_;
    this->index_offset_ = bisector.index_offset_;
  }
  return *this;
}

template <typename T>
ArrowBisector<T>& ArrowBisector<T>::operator=(ArrowBisector&& bisector) {
  if (this != &bisector) {  // not a self-assignment
    this->values_ = std::move(bisector.values_);
    this->sorted_indices_ = std::move(bisector.sorted_indices_);
    this->length_ = bisector.length_;
    this->indices_ = bisector.indices_;
    this->index_offset_ = bisector.index_offset_;

    bisector.length_ = 0;
    bisector.sorted_indices_ = nullptr;
  }
  return *this;
}

template <typename T>
inline int64_t ArrowBisector<T>::find(T const& v) const {
  // see also:
  //  - https://en.cppreference.com/w/cpp/algorithm/binary_search
  //  - https://en.cppreference.com/w/cpp/algorithm/lower_bound
  const uint64_t *first = indices_, *last = indices_ + length_;

  size_t count = length_, step = 0;
  while (count > 0) {
    step = count / 2;
    const uint64_t* it = first + step;
    if (values_->GetView(*it) < v) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  if (first != last && values_->GetView(*first) == v) {
    return (first - indices_) + index_offset_;
  } else {
    return -1;
  }
}

template <typename T>
inline bool ArrowBisector<T>::has(T const& v) const {
  return find(v) == -1;
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_VERTEX_MAP_ARROW_BISECTOR_IMPL_H_
