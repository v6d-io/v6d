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
#ifndef MODULES_FUSE_CACHE_MANAGER_MANAGER_H_
#define MODULES_FUSE_CACHE_MANAGER_MANAGER_H_
#include <iostream>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "arrow/buffer.h"
#include "common/util/logging.h"
#include "common/util/status.h"
namespace vineyard {
namespace fuse {

namespace cache_manager {
template <typename K, typename V>
struct KeyValue {
  using KeyType = K;
  using ValType = std::shared_ptr<V>;
  const KeyType key;
  ValType value;
  KeyValue(KeyType k, ValType v) : key(k), value(v) { ; }
  KeyValue(const KeyValue<K, V>& kv) : key(kv.key), value(kv.value) { ; }
};

// template<typename K, typename V>
// class CacheManager;
// using CMII = CacheManager<int,int>;

template <typename KeyValue>
class CacheManager {
 private:
  std::list<KeyValue> myList;
  std::unordered_map<typename KeyValue::KeyType,
                     typename std::list<KeyValue>::iterator>
      myMap;
  size_t capacityBytes;
  size_t curBytes;
  void popToNBytes(size_t n);
  bool WithInCapacity(size_t data);

 public:
  explicit CacheManager(size_t capacity);
  CacheManager();
  ~CacheManager();
  void resize(size_t capacity);
  Status put(const typename KeyValue::KeyType& key,
             typename KeyValue::ValType val);
  typename KeyValue::ValType get(const typename KeyValue::KeyType& key);
  bool has(const typename KeyValue::KeyType& key);
  std::list<KeyValue> getLinkedList();
  typename KeyValue::ValType operator[](const typename KeyValue::KeyType& key);
  size_t getCurBytes();
  size_t getCapacityBytes();
  void destroy();
};
}  // namespace cache_manager

}  // namespace fuse

};      // namespace vineyard
#endif  // MODULES_FUSE_CACHE_MANAGER_MANAGER_H_
