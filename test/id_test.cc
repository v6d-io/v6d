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

#include <bitset>
#include <cstdint>
#include <iostream>

#include "common/util/logging.h"
#include "common/util/uuid.h"

using vineyard::ObjectID;

int main() {
  // after revise
  ObjectID id1 = vineyard::GenerateBlobID(reinterpret_cast<uintptr_t>(&main));
  LOG(INFO) << id1 << "\n";
  CHECK(vineyard::IsBlob(id1));
  ObjectID id2 = vineyard::GenerateObjectID();
  LOG(INFO) << id2 << "\n";
  CHECK(!vineyard::IsBlob(id2));
}
