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

static void* __load_internal_registry(std::string& error_message) {
  void* handle = nullptr;
  auto lib = read_env("__VINEYARD_INTERNAL_REGISTRY");
  int dlflags = RTLD_GLOBAL | RTLD_NOW;
  if (!lib.empty()) {
    handle = dlopen(lib.c_str(), dlflags);
    if (handle == nullptr) {
      auto err = dlerror();
      if (err) {
        error_message = err;
#ifndef NDEBUG
        // See: Note [std::cerr instead of DVLOG()]
        std::cerr << "vineyard: error in loading: " << err << std::endl;
#endif
      }
    }
  }
  if (handle == nullptr) {
#if __APPLE__
    handle = dlopen("libvineyard_internal_registry.dylib", dlflags);
#else
    handle = dlopen("libvineyard_internal_registry.so", dlflags);
#endif
    if (handle == nullptr) {
      auto err = dlerror();
      if (err) {
        error_message = err;
#ifndef NDEBUG
        // See: Note [std::cerr instead of DVLOG()]
        std::cerr << "vineyard: error in loading from default: " << err
                  << std::endl;
#endif
      }
    }
  }
  return handle;
}

static vineyard_registry_getter_t __find_global_registry_entry(
    std::string& error_message) {
  auto ret = reinterpret_cast<vineyard_registry_getter_t>(
      dlsym(RTLD_DEFAULT, "__GetGlobalVineyardRegistry"));
  if (ret == nullptr) {
    auto err = dlerror();
    if (err) {
      error_message = err;
#ifndef NDEBUG
      // See: Note [std::cerr instead of DVLOG()]
      std::cerr << "vineyard: error in resolving: " << err << std::endl;
#endif
    }
  }
  return ret;
}

static std::unordered_map<std::string, ObjectFactory::object_initializer_t>*
__instantize__registry(vineyard_registry_handler_t& handler,
                       vineyard_registry_getter_t& getter) {
  if (!read_env("VINEYARD_USE_LOCAL_REGISTRY").empty()) {
    return new std::unordered_map<std::string, ObjectFactory::object_initializer_t>();
  }

  if (getter == nullptr) {
    std::string error_message;

    // load from the global scope
#ifndef NDEBUG
    // See: Note [std::cerr instead of DVLOG()]
    std::cerr << "vineyard: looking up vineyard registry from the global scope"
              << std::endl;
#endif
    getter = detail::__find_global_registry_entry(error_message);

    // load the shared library, then search from the global scope
    if (getter == nullptr) {
#ifndef NDEBUG
      // See: Note [std::cerr instead of DVLOG()]
      std::cerr << "vineyard: loading the vineyard registry library"
                << std::endl;
#endif
      handler = detail::__load_internal_registry(error_message);
      VINEYARD_ASSERT(handler != nullptr,
                      "Failed to load the vineyard global registry registry: " +
                          error_message);

      // load from the global scope, again
#ifndef NDEBUG
      // See: Note [std::cerr instead of DVLOG()]
      std::cerr << "vineyard: looking up vineyard registry from the global "
                   "scope again"
                << std::endl;
#endif
      getter = detail::__find_global_registry_entry(error_message);
    }

    VINEYARD_ASSERT(getter != nullptr,
                    "Failed to load the vineyard global registry entries: " +
                        error_message);
  }

  return reinterpret_cast<
      std::unordered_map<std::string, ObjectFactory::object_initializer_t>*>(getter());
}

}  // namespace detail

std::unordered_map<std::string, ObjectFactory::object_initializer_t>&
ObjectFactory::getKnownTypes() {
  static std::unordered_map<std::string, object_initializer_t>*
      __internal__registry = detail::__instantize__registry(
          __registry_handle, __GetGlobalRegistry);
  return *__internal__registry;
}

vineyard_registry_handler_t ObjectFactory::__registry_handle = nullptr;
vineyard_registry_getter_t ObjectFactory::__GetGlobalRegistry = nullptr;

}  // namespace vineyard
