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

#include <dlfcn.h>

#include "client/ds/object_factory.h"

#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "common/util/env.h"

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

std::unique_ptr<Object> ObjectFactory::Create(ObjectMeta const& metadata) {
  return ObjectFactory::Create(metadata.GetTypeName(), metadata);
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

namespace detail {

static void* __load_internal_registry() {
  void* handle = nullptr;
  auto lib = read_env("__VINEYARD_INTERNAL_REGISTRY");
  if (!lib.empty()) {
    handle = dlopen(lib.c_str(), RTLD_GLOBAL | RTLD_LAZY);
  }
  if (handle == nullptr) {
#if __APPLE__
    handle =
        dlopen("libvineyard_internal_registry.dylib", RTLD_GLOBAL | RTLD_LAZY);
#else
    handle =
        dlopen("libvineyard_internal_registry.so", RTLD_GLOBAL | RTLD_LAZY);
#endif
  }
  return handle;
}

static vineyard_registry_getter_t __find_global_registry_entry() {
  return reinterpret_cast<vineyard_registry_getter_t>(
      dlsym(RTLD_DEFAULT, "__GetGlobalVineyardRegistry"));
}

}  // namespace detail

std::unordered_map<std::string, ObjectFactory::object_initializer_t>&
ObjectFactory::getKnownTypes() {
  if (__GetGlobalRegistry == nullptr) {
    // load from the global scope
    DVLOG(10) << "Looking up vineyard registry from the global scope";
    __GetGlobalRegistry = detail::__find_global_registry_entry();

    // load the shared library, then search from the global scope
    if (__GetGlobalRegistry == nullptr) {
      DVLOG(10) << "Loading the vineyard registry library";
      __registry_handle = detail::__load_internal_registry();
    }
    VINEYARD_ASSERT(__registry_handle != nullptr,
                    "Failed to load the vineyard global registry registry: " +
                        std::string(dlerror()));

    // load from the global scope, again
    DVLOG(10) << "Looking up vineyard registry from the global scope again";
    __GetGlobalRegistry = detail::__find_global_registry_entry();
  }
  VINEYARD_ASSERT(__GetGlobalRegistry != nullptr,
                  "Failed to load the vineyard global registry entries");

  static std::unordered_map<std::string,
                            object_initializer_t>* __internal__registry =
      reinterpret_cast<std::unordered_map<std::string, object_initializer_t>*>(
          __GetGlobalRegistry());

  return *__internal__registry;
}

void* ObjectFactory::__registry_handle = nullptr;
vineyard_registry_getter_t ObjectFactory::__GetGlobalRegistry = nullptr;

}  // namespace vineyard
