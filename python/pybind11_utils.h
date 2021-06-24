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

#ifndef PYBIND_11_UTILS_H__
#define PYBIND_11_UTILS_H__

#include <functional>
#include <sstream>

#include "pybind11/pybind11.h"

#pragma GCC visibility push(default)
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#pragma GCC visibility pop

namespace vineyard {

// Wrap ObjectID to makes pybind11 work.
struct ObjectIDWrapper {
  ObjectIDWrapper() : internal_id(InvalidObjectID()) {}
  ObjectIDWrapper(ObjectID id) : internal_id(id) {}
  explicit ObjectIDWrapper(std::string const& id)
      : internal_id(VYObjectIDFromString(id)) {}
  explicit ObjectIDWrapper(const char* id)
      : internal_id(VYObjectIDFromString(id)) {}
  operator ObjectID() const { return internal_id; }
  bool operator==(ObjectIDWrapper const& other) const {
    return internal_id == other.internal_id;
  }

 private:
  ObjectID internal_id;
};

// Wrap ObjectName to makes pybind11 work.
struct ObjectNameWrapper {
  explicit ObjectNameWrapper(std::string const& name) : internal_name(name) {}
  explicit ObjectNameWrapper(const char* name) : internal_name(name) {}
  operator std::string() const { return internal_name; }
  bool operator==(ObjectNameWrapper const& other) const {
    return internal_name == other.internal_name;
  }

 private:
  const std::string internal_name;
};

}  // namespace vineyard

namespace pybind11 {

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator, typename Sentinel, typename... Extra>
iterator make_iterator_fmap(Iterator first, Sentinel last,
                            std::function<object(Iterator&)> functor,
                            Extra&&... extra) {
  typedef detail::iterator_state<Iterator, Sentinel, false, Policy> state;

  if (!detail::get_type_info(typeid(state), false)) {
    class_<state>(handle(), "iterator", pybind11::module_local())
        .def("__iter__", [](state& s) -> state& { return s; })
        .def(
            "__next__",
            [functor](state& s) -> object {
              if (!s.first_or_done)
                ++s.it;
              else
                s.first_or_done = false;
              if (s.it == s.end) {
                s.first_or_done = true;
                throw stop_iteration();
              }
              return functor(s.it);
            },
            std::forward<Extra>(extra)..., Policy);
  }

  return cast(state{first, last, true});
}

}  // namespace pybind11

namespace vineyard {

void throw_on_error(Status const& status);

pybind11::object json_to_python(json const& value);

}  // namespace vineyard

#endif  // PYBIND_11_UTILS_H__
