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
#include <unordered_map>
#include <vector>

#include "server/memory/usage.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using LRU =
    detail::ColdObjectTracker<uint64_t, std::string, decltype(nullptr)>::LRU;

void BasicTest() {
  LRU lru_;
  std::vector<uint64_t> ids;
  std::vector<std::shared_ptr<std::string>> payloads;
  ids.reserve(1000);
  payloads.reserve(1000);
  {
    // insert
    for (int i = 0; i < 1000; i++) {
      ids.push_back(i);
      payloads.push_back(std::make_shared<std::string>(std::to_string(i)));
      lru_.Ref(ids.back(), payloads.back());
    }
  }
  {
    for (int i = 0; i < 1000; i++) {
      CHECK(lru_.CheckExist(i));
    }
  }
}

int main(int argc, char** argv) {
  BasicTest();
  LOG(INFO) << "Passed lru tests...";
  return 0;
}
