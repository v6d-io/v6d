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

#include <dlfcn.h>
#include <libgen.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <libproc.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <unistd.h>
#endif

#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "common/util/env.h"

namespace vineyard {

std::unique_ptr<Object> ObjectFactory::Create(std::string const& type_name) {
  auto& known_types = getKnownTypes();
  auto creator = known_types.find(type_name);
  if (creator == known_types.end()) {
#ifndef NDEBUG
    static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
    if (__trace) {
      std::clog << "[debug] create an instance with the unknown typename: "
                << type_name << std::endl;
    }

#endif
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
#ifndef NDEBUG
    static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
    if (__trace) {
      std::clog << "[debug] create an instance with the unknown typename: "
                << type_name << std::endl;
    }
#endif
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

static void* __try_load_internal_registry(std::string const& location,
                                          std::string& error_message) {
  if (location.empty()) {
    return nullptr;
  }
  int dlflags = RTLD_GLOBAL | RTLD_NOW;
  void* handle = dlopen(location.c_str(), dlflags);
  if (handle == nullptr) {
    auto err = dlerror();
    if (err) {
      error_message = err;
#ifndef NDEBUG
      static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
      if (__trace) {
        // See: Note [std::cerr instead of DVLOG()]
        std::clog << "vineyard: error in loading: " << err << std::endl;
      }
#endif
    }
  }
  return handle;
}

extern "C" {
int __find_vineyard_library_location(Dl_info* info) {
  // void *fn = *reinterpret_cast<void**>(&__find_vineyard_library_location);
  void* fn = reinterpret_cast<void*>(&__find_vineyard_library_location);
  return dladdr(fn, info);
}
}

static void* __load_internal_registry(std::string& error_message) {
  void* handle = nullptr;

  // search from environment variable
  const std::string lib_from_env = read_env("__VINEYARD_INTERNAL_REGISTRY");
  if (access(lib_from_env.c_str(), F_OK) == 0) {
    handle = __try_load_internal_registry(lib_from_env, error_message);
  }
  if (handle != nullptr) {
    return handle;
  }

  // search from the current location where `vineyard_client` locates
  Dl_info info;
  if (__find_vineyard_library_location(&info) != 0) {
    char* fname = strndup(info.dli_fname, PATH_MAX);
    if (fname != nullptr && info.dli_fname[0] != '\0') {
#if __APPLE__
      handle = __try_load_internal_registry(
          std::string(dirname(fname)) + "/libvineyard_internal_registry.dylib",
          error_message);
#else
      handle = __try_load_internal_registry(
          std::string(dirname(fname)) + "/libvineyard_internal_registry.so",
          error_message);
#endif
    }
    if (fname != nullptr) {
      free(fname);
    }
  }
  if (handle != nullptr) {
    return handle;
  }

  // search from the LD_LIBRARY_PATH
#if __APPLE__
  handle = __try_load_internal_registry("libvineyard_internal_registry.dylib",
                                        error_message);
#else
  handle = __try_load_internal_registry("libvineyard_internal_registry.so",
                                        error_message);
#endif
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
      static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
      if (__trace) {
        // See: Note [std::clog instead of DVLOG()]
        std::clog << "vineyard: error in resolving: " << err << std::endl;
      }
#endif
    }
  }
  return ret;
}

static std::unordered_map<std::string, ObjectFactory::object_initializer_t>*
__instantize__registry(vineyard_registry_handler_t& handler,
                       vineyard_registry_getter_t& getter) {
  if (getter == nullptr) {
    std::string error_message;

    // load from the global scope
#ifndef NDEBUG
    static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
    if (__trace) {
      // See: Note [std::clog instead of DVLOG()]
      std::clog
          << "vineyard: looking up vineyard registry from the global scope"
          << std::endl;
    }
#endif
    getter = detail::__find_global_registry_entry(error_message);

    // load the shared library, then search from the global scope
    if (getter == nullptr) {
#ifndef NDEBUG
      if (__trace) {
        // See: Note [std::clog instead of DVLOG()]
        std::clog << "vineyard: loading the vineyard registry library"
                  << std::endl;
      }
#endif
      handler = detail::__load_internal_registry(error_message);
      VINEYARD_ASSERT(handler != nullptr,
                      "Failed to load the vineyard global registry registry: " +
                          error_message);

      // load from the global scope, again
#ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      if (__trace) {
        std::clog << "vineyard: looking up vineyard registry from the global "
                     "scope again"
                  << std::endl;
      }
#endif
      getter = detail::__find_global_registry_entry(error_message);
    }

    VINEYARD_ASSERT(getter != nullptr,
                    "Failed to load the vineyard global registry entries: " +
                        error_message);
  }

  auto registry = reinterpret_cast<
      std::unordered_map<std::string, ObjectFactory::object_initializer_t>*>(
      getter());

  if (!read_env("VINEYARD_USE_LOCAL_REGISTRY").empty()) {
    return new std::unordered_map<std::string,
                                  ObjectFactory::object_initializer_t>();
  }
  return registry;
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
