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

#ifndef MODULES_BASIC_UTILS_H_
#define MODULES_BASIC_UTILS_H_

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace vineyard {

template <typename ITER_T, typename FUNC_T>
void parallel_for(
    const ITER_T& begin, const ITER_T& end, const FUNC_T& func,
    const size_t parallelism = std::thread::hardware_concurrency(),
    size_t chunk = 0) {
  std::vector<std::thread> threads(parallelism);
  size_t num = end - begin;
  if (chunk == 0) {
    chunk = (num + parallelism - 1) / parallelism;
  }
  std::atomic<size_t> cur(0);
  for (size_t thread_index = 0; thread_index < parallelism; ++thread_index) {
    threads[thread_index] = std::thread([&]() {
      while (true) {
        size_t x = cur.fetch_add(chunk);
        if (x >= num) {
          break;
        }
        size_t y = std::min(x + chunk, num);
        ITER_T a = begin + x;
        ITER_T b = begin + y;
        while (a != b) {
          func(a);
          ++a;
        }
      }
    });
  }
  for (auto& thrd : threads) {
    thrd.join();
  }
}
}  // namespace vineyard

#endif  // MODULES_BASIC_UTILS_H_
