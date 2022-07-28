#include <unordered_map>
#include <dlfcn.h>

#include "adaptor_factory.h"
#include "common/util/status.h"

namespace vineyard {
namespace fuse {






namespace detail {

/* static void* __try_load_internal_registry(std::string const& location,
                                          std::string& error_message) {
              std::clog << "__try_load_internal_registry  is called"<< std::endl;
#ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "vineyard::fuse::detail::__try_load_internal_registry is called " << err << std::endl;
#endif
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
      // See: Note [std::cerr instead of DVLOG()]
      std::clog << "vineyard: error in loading: " << err << std::endl;
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
#ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "vineyard::fuse::detail::__load_internal_registry is called " << std::endl;
#endif
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
} */

static vineyard_deserializers_getter_t __find_global_fuse_adaptors_registry_entry(
    std::string& error_message) {
      #ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "vineyard::fuse::detail::__find_global_fuse_adaptors_registry_entry is called " << std::endl;
#endif
  auto ret = reinterpret_cast<vineyard_deserializers_getter_t>(
      dlsym(RTLD_DEFAULT, "__CreateGlobalVineyardFuseAdaptorsRegistry"));
  if (ret == nullptr) {
    auto err = dlerror();
    if (err) {
      error_message = err;
#ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "vineyard: error in resolving: " << err << std::endl;
#endif
    }
  }
  std::clog << "vineyard::fuse::detail::__find_global_fuse_adaptors_registry_entry exited " << std::endl;

  return ret;
}

static std::unordered_map<std::string, vineyard_deserializer_t>*
__instantize__registry(vineyard_deserializers_getter_t& getter) {
  #ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "vineyard::fuse::detail::__instantize__registry is called "  << std::endl;
#endif
  if (getter == nullptr) {
    std::string error_message;
    // load from the global scope
#ifndef NDEBUG
    // See: Note [std::clog instead of DVLOG()]
    std::clog << "vineyard: looking up vineyard registry from the global scope"
              << std::endl;
#endif
    getter = detail::__find_global_fuse_adaptors_registry_entry(error_message);

    // load the shared library, then search from the global scope
//     if (getter == nullptr) {
// #ifndef NDEBUG
//       // See: Note [std::clog instead of DVLOG()]
//       std::clog << "vineyard: loading the vineyard registry library"
//                 << std::endl;
// #endif
//       handler = detail::__load_internal_registry(error_message);
//       VINEYARD_ASSERT(handler != nullptr,
//                       "Failed to load the vineyard global registry registry: " +
//                           error_message);

      // load from the global scope, again
// #ifndef NDEBUG
//       // See: Note [std::clog instead of DVLOG()]
//       std::clog << "vineyard: looking up vineyard registry from the global "
//                    "scope again"
//                 << std::endl;
// #endif
//       getter = detail::__find_global_registry_entry(error_message);
    // }

    VINEYARD_ASSERT(getter != nullptr,
                    "Failed to load the vineyard global registry entries: " +
                        error_message);
  }

  auto registry = reinterpret_cast<
      std::unordered_map<std::string, fuse::vineyard_deserializer_t>*>(
      getter());

  // if (!read_env("VINEYARD_USE_LOCAL_REGISTRY").empty()) {
  //   return new std::unordered_map<std::string,
  //                                 ObjectFactory::object_initializer_t>();
  // }
  return registry;
}

}  // namespace detail



std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_t>&
   AdaptorFactory::getDeserializers()
 {
#ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "vineyard::fuse::AdaptorFactory::getDeserializers is called "  << std::endl;
#endif
  static std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_t>*
      __internal__registry = vineyard::fuse::detail::__instantize__registry(
          vineyard::fuse::AdaptorFactory::__GetGlobalVineyardFuseAdaptorsRegistry);
  return *__internal__registry;  
}

    






vineyard_deserializers_getter_t AdaptorFactory::__GetGlobalVineyardFuseAdaptorsRegistry = nullptr;


}  // namespace fuse
}  // namespace vineyard
