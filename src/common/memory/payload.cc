/** Copyright 2020 Alibaba Group Holding Limited.

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

#include "common/memory/payload.h"

namespace vineyard {

void Payload::ToJSON(ptree& tree) const {
  tree.put("object_id", object_id);
  tree.put("store_fd", store_fd);
  tree.put("data_offset", data_offset);
  tree.put("data_size", data_size);
  tree.put("map_size", map_size);
}

void Payload::FromJSON(const ptree& tree) {
  object_id = tree.get<ObjectID>("object_id");
  store_fd = tree.get<int>("store_fd");
  data_offset = tree.get<ptrdiff_t>("data_offset");
  data_size = tree.get<int64_t>("data_size");
  map_size = tree.get<int64_t>("map_size");
  pointer = nullptr;
}

}  // namespace vineyard
