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

#ifndef SRC_CLIENT_DS_OBJECT_FACTORY_H_
#define SRC_CLIENT_DS_OBJECT_FACTORY_H_

#include <dlfcn.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/util/env.h"
#include "common/util/typename.h"

namespace vineyard {

/** Note [std::clog instead of DVLOG()]
 *
 * In the object factory we use `std::clog` instead of `DVLOG()` in glog for
 * logging, as, the logging happens during initializing the shared library,
 * (in the static field), at that stage the data structure of glog hasn't
 * been initialized yet, leading to crash.
 */

class Client;
class Object;
class ObjectMeta;

/**
 * @brief FORCE_INSTANTIATE is a tool to guarantee the argument not be optimized
 * by the compiler, even when it is unused. This trick is useful when we want to
 * hold a reference of static member in constructors.
 */
template <typename T>
inline void FORCE_INSTANTIATE(T) {}

using vineyard_registry_handler_t = void*;
using vineyard_registry_getter_t = void* (*) ();

/**
 * @brief ObjectFactory is responsible for type registration at the
 * initialization time.
 */
class ObjectFactory {
 public:
  using object_initializer_t = std::unique_ptr<Object> (*)();

  /**
   * @brief Register the type `T` to the factory. A registrable type must have
   * a static method `Create` that accepts no arguments and returns an instance
   * of type `T`.
   *
   * @tparam T The concrete to be registered to vineyard's object type
   * resolution.
   */
  template <typename T>
  static bool Register() {
    const std::string name = type_name<T>();
#ifndef NDEBUG
    static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
    if (__trace) {
      // See: Note [std::cerr instead of DVLOG()]
      std::clog << "vineyard: register data type: " << name << std::endl;
    }
#endif
    auto& known_types = getKnownTypes();
    // the explicit `static_cast` is used to help overloading resolution.
    known_types[name] = static_cast<object_initializer_t>(&T::Create);
    return true;
  }

  /**
   * @brief Initialize an instance by looking up the `type_name` in the factory.
   *
   * @param type_name The type to be instantiated.
   */
  static std::unique_ptr<Object> Create(std::string const& type_name);

  /**
   * @brief Initialize an instance by looking up the `type_name` in the factory,
   * and construct the object using the metadata.
   *
   * @param metadata The metadata used to construct the object.
   */
  static std::unique_ptr<Object> Create(ObjectMeta const& metadata);

  /**
   * @brief Initialize an instance by looking up the `type_name` in the factory,
   * and construct the object using the metadata.
   *
   * We keep this variant with explicit `typename` for fine-grained control of
   * the resolver.
   *
   * @param type_name The type to be instantiated.
   * @param metadata The metadata used to construct the object.
   */
  static std::unique_ptr<Object> Create(std::string const& type_name,
                                        ObjectMeta const& metadata);

  /**
   * @brief Expose the internal registered types.
   *
   * @return A map of type name to that type's static constructor.
   */
  static const std::unordered_map<std::string, object_initializer_t>&
  FactoryRef();

 private:
  // https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static std::unordered_map<std::string, object_initializer_t>& getKnownTypes();

  static vineyard_registry_handler_t __registry_handle;
  static vineyard_registry_getter_t __GetGlobalRegistry;
};

}  // namespace vineyard

namespace std {

// std::unique_ptr casts:
template <class T, class U>
inline unique_ptr<T> static_pointer_cast(unique_ptr<U>&& r) noexcept {
  return std::unique_ptr<T>(static_cast<T*>(r.release()));
}

template <class T, class U>
inline unique_ptr<T> dynamic_pointer_cast(unique_ptr<U>&& r) noexcept {
  return std::unique_ptr<T>(dynamic_cast<T*>(r.release()));
}

template <class T, class U>
inline unique_ptr<T> const_pointer_cast(unique_ptr<U>&& r) noexcept {
  return std::unique_ptr<T>(const_cast<T*>(r.release()));
}

template <class T, class U>
inline unique_ptr<T> reinterpert_pointer_cast(unique_ptr<U>&& r) noexcept {
  return std::unique_ptr<T>(reinterpret_cast<T*>(r.release()));
}
}  // namespace std

#endif  // SRC_CLIENT_DS_OBJECT_FACTORY_H_
