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
#include <sstream>

#include "basic/ds/arrow.h"
#include "basic/ds/hashmap.h"
#include "common/util/logging.h"
#include "common/util/typename.h"
#include "modules/fuse/cache_manager/manager.h"
#include "modules/fuse/cache_manager/manager.hpp"
namespace vfc = vineyard::fuse::cache_manager;

vfc::KeyValue<std::string, arrow::Buffer> gkv(int64_t len) {
  static const char alpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  arrow::BufferBuilder b;
  b.Resize(len);
  for (int i = 0; i < len; ++i) {
    b.Append(1, alpha[i % 26]);
  }
  std::shared_ptr<arrow::Buffer> p;
  b.Finish(&p);
  return vfc::KeyValue<std::string, arrow::Buffer>(std::to_string(len), p);
}

int main() {
  size_t max_cacpacity = 20;  // bytes
  vfc::CacheManager<vfc::KeyValue<std::string, arrow::Buffer>> cm(
      max_cacpacity);

  {
    vfc::CacheManager<vfc::KeyValue<std::string, arrow::Buffer>> cm(
        max_cacpacity);

    // put get
    auto kv = gkv(10);
    DLOG(INFO) << kv.value->ToString();
    cm.put(kv.key, kv.value);
    cm.get(kv.key);
  }
  // get non-existent file
  {
    vfc::CacheManager<vfc::KeyValue<std::string, arrow::Buffer>> cm(
        max_cacpacity);

    auto kv = gkv(9);
    if (cm.get(kv.key) != nullptr) {
      DLOG(INFO) << "it surprisingly get non-existent object" << std::endl;
    }
  }
  // get file
  // put file that is greater than the compacity
  {
    vfc::CacheManager<vfc::KeyValue<std::string, arrow::Buffer>> cm(
        max_cacpacity);

    auto kv = gkv(400);
    auto status = cm.put(kv.key, kv.value);
    if (status.ok())
      DLOG(INFO) << "it surprisingly accept the big file" << std::endl;
  }
  // pop the least recently used file
  {
    vfc::CacheManager<vfc::KeyValue<std::string, arrow::Buffer>> cm(200);

    for (int i = 1; i < 10; ++i) {
      {
        auto kv = gkv(i);
        DLOG(INFO) << "cache manager current bytes: " << cm.getCurBytes()
                   << " capacity: " << cm.getCapacityBytes();
        VINEYARD_CHECK_OK(cm.put(kv.key, kv.value));
        CHECK_LE(cm.getCurBytes(), cm.getCapacityBytes());
      }
    }
  }
  // maintain the odd numbered file
  {
    vfc::CacheManager<vfc::KeyValue<std::string, arrow::Buffer>> cm(200);

    for (int i = 1; i < 10; ++i) {
      auto kv = gkv(i);

      DLOG(INFO) << "cache manager current bytes: " << cm.getCurBytes()
                 << " capacity: " << cm.getCapacityBytes();
      VINEYARD_CHECK_OK(cm.put(kv.key, kv.value));
      for (int j = i; j >= 1; --j) {
        if (j % 2 == 1)
          cm.get(std::to_string(j));
      }
    }
    auto l = cm.getLinkedList();
    std::stringstream ss;
    for (auto i : l) {
      ss << " " << i.key;
      CHECK_DOUBLE_EQ(stoi(i.key) % 2, 1);
    }
    // std::cout<<ss.str()<<std::endl;
    DLOG(INFO) << "remaining key" << ss.str() << std::endl;
  }
}
