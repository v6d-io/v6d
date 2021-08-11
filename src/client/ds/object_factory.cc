/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"

namespace vineyard {

std::unique_ptr<Object> ObjectFactory::Create(std::string const& type_name) {
  auto& known_types = getKnownTypes();
  auto creator = known_types.find(type_name);
  if (creator == known_types.end()) {
    VLOG(11) << "Failed to create an instance due to the unknown typename: "
             << type_name;
    return nullptr;
  } else {
    return (creator->second)();
  }
}

std::unique_ptr<Object> ObjectFactory::Create(std::string const& type_name,
                                              ObjectMeta const& metadata) {
  auto& known_types = getKnownTypes();
  auto creator = known_types.find(type_name);
  if (creator == known_types.end()) {
    VLOG(11) << "Failed to create an instance due to the unknown typename: "
             << type_name;
    return nullptr;
  } else {
    auto target = (creator->second)();
    target->Construct(metadata);
    return target;
  }
}

const std::unordered_map<std::string, ObjectFactory::object_initializer_t>&
ObjectFactory::FactoryRef() {
  return getKnownTypes();
}

std::unordered_map<std::string, ObjectFactory::object_initializer_t>&
ObjectFactory::getKnownTypes() {
  static std::unordered_map<std::string, object_initializer_t>* known_types =
      new std::unordered_map<std::string, object_initializer_t>();
  return *known_types;
}

}  // namespace vineyard
