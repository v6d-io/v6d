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

#ifndef MODULES_GRAPH_UTILS_PERFECT_HASH_H_
#define MODULES_GRAPH_UTILS_PERFECT_HASH_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "graph/utils/BooPHF.h"

namespace vineyard{

#define IS_INTEGER_TYPE(K) (std::is_same<K, uint32_t>::value || std::is_same<K, int32_t>::value || std::is_same<K, int64_t>::value || std::is_same<K, uint64_t>::value)

template<typename K, typename V>
struct PerfectEntry {
  PerfectEntry() = default;
  PerfectEntry(std::pair<K, V> value) : value_(value) {}

  std::pair<K, V> value_;
};

template<typename K, typename V, typename E = std::equal_to<K>>
class PerfectHash{
  typedef boomphf::SingleHashFunctor<K> hasher_t;

public:
  PerfectHash() = default;

  PerfectHash(std::vector<K> &vec_k, std::vector<V> &vec_v, int concurrency = 1) {
    // LOG(INFO) << __func__ << " is not finished.";

    // auto data_iterator = boomphf::range(static_cast<const K*>(vec_k.data()), static_cast<const K*>(vec_k.data() + vec_k.size()));
    // bphf = boomphf::mphf<K, hasher_t>(vec_k.size(), data_iterator, concurrency, 1.0f);

    // vec_v_.resize(vec_v.size());
    // for (size_t i = 0; i < vec_v.size(); ++i) {
    //   vec_v_[bphf.lookup(vec_k[i])] = vec_v[i];
    //   LOG(INFO) << "origin k:" << vec_k[i] << " perf k:" << bphf.lookup(vec_k[i]) << " value:" << vec_v[i] << " i:" << i;
    //   entries_.emplace_back(PerfectEntry(std::pair<K, V>(vec_k[i], vec_v[i])));
    // }
  }

  template<typename K_ = K>
  typename std::enable_if<IS_INTEGER_TYPE(K_), void>::type
  Construct(int concurrency = 1) {
    size_t count = 0;
    vec_k_.resize(vec_kv_.size());
    for (auto &kv_ : vec_kv_) {
      vec_k_[count] = kv_.first;
      count++;
    }
    auto data_iterator = boomphf::range(vec_k_.begin(), vec_k_.end());
    bphf = boomphf::mphf<K, hasher_t>(vec_k_.size(), data_iterator, concurrency, 1.0f);

    vec_v_.resize(count);
    for (size_t i = 0; i < vec_v_.size(); i++) {
      vec_v_[bphf.lookup(vec_k_[i])] = vec_kv_[i].second;
    }
    // entries_.resize(count);
    // for (size_t i = 0; i < entries_.size(); i++) {
    //   entries_[bphf.lookup(vec_kv_[i].first)] = PerfectEntry(vec_kv_[i].first, vec_kv_[i].second);
    // }
  }

  template<typename K_ = K>
  typename std::enable_if<!IS_INTEGER_TYPE(K_), void>::type
  Construct(int concurrency = 1) {
    LOG(INFO) << __func__ << " is not supported yet.";
  }

  void clear() {
    LOG(INFO) << __func__ << " is not finished.";
  }

  size_t size() {
    return vec_kv_.size();
  }

  bool empty() {
    LOG(INFO) << __func__ << " is not finished.";
    return true;
  }

  int find(const K &key) {
    LOG(INFO) << __func__ << " is not finished.";
    return -1;
  }

  int begin() {
    LOG(INFO) << __func__ << " is not finished.";
    return -1;
  }

  int end() {
    LOG(INFO) << __func__ << " is not finished.";
    return -1;
  }

  void reserve(size_t size) {
    LOG(INFO) << __func__ << " is not finished.";
    // vec_kv_.resize(size);
  }

  void emplace(const K &key, const V &value) {
    // LOG(INFO) << __func__ << " is not finished.";
    //TODO : check if the vertex add twice.
    vec_kv_.push_back(std::pair<K, V>(key, value));
    // LOG(INFO) << "key:"  << key << " value:" << value;
  }

  inline V & operator[](const K & key) {
    V v;
    LOG(INFO) << __func__ << " is not finished.";
    return *(new V());
  }

 public:
  boomphf::mphf<K, hasher_t> bphf;
  std::vector<V> vec_v_;
  std::vector<K> vec_k_;
  std::vector<PerfectEntry<K, V>> entries_;
  std::vector<std::pair<K, V>> vec_kv_;
};

} // namespace vineyard

#endif // MODULES_GRAPH_UTILS_PERFECT_HASH_H_