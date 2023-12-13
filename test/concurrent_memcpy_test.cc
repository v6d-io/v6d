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
#include <string>
#include <thread>

#include "common/memory/memcpy.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void testing_memcpy(const size_t size) {
  LOG(INFO) << "Testing memcpy with size: " << size;
  std::unique_ptr<char[]> src(new char[size]);
  std::unique_ptr<char[]> dst(new char[size]);
  for (size_t i = 0; i < size; ++i) {
    src[i] = i % 256;
  }

  std::vector<size_t> sizes_to_test = {
      size, size - 1, size - 3, size - 10, size - 1024, size - 1024 * 1024,
  };

  for (size_t size_to_test : sizes_to_test) {
    if (size_to_test > size) {
      continue;
    }
    for (size_t concurrency = 1; concurrency <= 8; ++concurrency) {
      memset(dst.get(), 0, size_to_test);
      memory::concurrent_memcpy(dst.get(), src.get(), size_to_test,
                                concurrency);
      CHECK_EQ(0, memcmp(dst.get(), src.get(), size_to_test));
    }
  }
  LOG(INFO) << "Passed memcpy test with size: " << size;
}

int main(int argc, char** argv) {
  if (argc < 1) {
    printf("usage ./concurrent_memcpy_test");
    return 1;
  }

  for (size_t sz = 1024 * 1024 * 8; sz < 1024 * 1024 * 1024;
       sz += 1024 * 1024 * 256) {
    testing_memcpy(sz);
  }

  LOG(INFO) << "Passed concurrent memcpy tests...";

  return 0;
}
