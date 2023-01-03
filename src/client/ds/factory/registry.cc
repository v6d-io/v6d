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

#include "client/ds/object_factory.h"

extern "C" __attribute__((visibility("default"))) void*
__GetGlobalVineyardRegistry() {
  static std::unordered_map<
      std::string, vineyard::ObjectFactory::object_initializer_t>* known_types =
      new std::unordered_map<std::string,
                             vineyard::ObjectFactory::object_initializer_t>();
  return reinterpret_cast<void*>(known_types);
}
