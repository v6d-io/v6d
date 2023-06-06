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

#include <thread>

namespace vineyard {

template <typename ITER_T, typename FUNC_T>
void parallel_for(const ITER_T& begin, const ITER_T& end, const FUNC_T& func,
                  int thread_num, size_t chunk = 0) {
  std::vector<std::thread> threads(thread_num);
  size_t num = end - begin;
  if (chunk == 0) {
    chunk = (num + thread_num - 1) / thread_num;
  }
  std::atomic<size_t> cur(0);
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread([&]() {
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