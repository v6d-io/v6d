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

#ifndef SRC_CLIENT_DS_OBJECT_FACTORY_H_
#define SRC_CLIENT_DS_OBJECT_FACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "common/util/logging.h"
#include "common/util/typename.h"

namespace vineyard {

class Client;
class Object;

/**
 * @brief FORCE_INSTANTIATE is a tool to guarantee the argument not be optimized
 * by the compiler, even when it is unused. This trick is useful when we want
 * hold a reference of static member in constructors.
 */
template <typename T>
inline void FORCE_INSTANTIATE(T) {}

/**
 * @brief ObjectFactory is responsible for type registration at the
 * initialization time.
 */
class ObjectFactory {
 public:
  using object_initializer_t = std::shared_ptr<Object> (*)();

  /**
   * @brief Register tht type `T` to the factory. A registrable type must have
   * a static method `Create` that accepts no arguments and returns an instance
   * of type `T`.
   *
   * @tparam T The concrete to be registered to vineyard's object type
   * resolution.
   */
  template <typename T>
  static bool Register() {
    auto& known_types = getKnownTypes();
    // the explicit `static_cast` is used to help overloading resolution.
    known_types.emplace(type_name<T>(),
                        static_cast<object_initializer_t>(&T::Create));
    return true;
  }

  /**
   * @brief Initialize an instance by looking up the `type_name` in the factory.
   *
   * @param type_name The type to be instantiated.
   */
  static std::shared_ptr<Object> Create(std::string const& type_name);

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
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_OBJECT_FACTORY_H_
