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

#include <memory>
#include <vector>

#include "kv-state-cache/radix-tree/radix-tree.h"

#ifndef MODULES_CACHE_STRATEGY_H_
#define MODULES_CACHE_STRATEGY_H_

namespace vineyard {

class CacheStrategy {
 protected:
  int capacity;

 public:
  /**
   * @brief put a key into the cache
   * @param key the key to be put
   * @return the key to be evicted if the cache is full, otherwise return empty
   * vector
   */

  virtual void put(const std::vector<int>& prefix, int token,
                   std::vector<int>& evicted_tokens) = 0;
};

}  // namespace vineyard

#endif