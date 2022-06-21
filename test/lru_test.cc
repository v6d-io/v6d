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

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "server/memory/usage.h"

#define LRU \
  detail::ColdObjectTracker<uint64_t, std::string, decltype(nullptr)>::LRU

using namespace vineyard;  // NOLINT
using namespace std;       // NOLINT

void BasicTest() {
  LRU lru_;
  vector<uint64_t> ids;
  vector<shared_ptr<string>> payloads;
  ids.reserve(1000);
  payloads.reserve(1000);
  {
    // insert
    for (int i = 0; i < 1000; i++) {
      ids.push_back(i);
      payloads.push_back(make_shared<string>(to_string(i)));
      lru_.Ref(ids.back(), payloads.back());
    }
  }
  {
    for (int i = 0; i < 1000; i++) {
      CHECK(lru_.CheckExist(i));
    }
  }
  {
    // Pop and check if id pops from 999 to 0
    for (int i = 0; i < 1000; i++) {
      auto ret = lru_.PopLeastUsed();
      CHECK(ret.first.ok());
      CHECK(ret.second.first == static_cast<uint64_t>(i));
    }
  }
}

int main(int argc, char** argv) {
  BasicTest();
  LOG(INFO) << "Passed lru tests...";
  return 0;
}
