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

#ifndef PYTHON_PYBIND11_UTILS_H_
#define PYTHON_PYBIND11_UTILS_H_

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "common/memory/memcpy.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace vineyard {

// Wrap ObjectID to makes pybind11 work.
struct ObjectIDWrapper {
  ObjectIDWrapper() : internal_id(InvalidObjectID()) {}
  ObjectIDWrapper(ObjectID id) : internal_id(id) {}  // NOLINT(runtime/explicit)
  explicit ObjectIDWrapper(std::string const& id)
      : internal_id(ObjectIDFromString(id)) {}
  explicit ObjectIDWrapper(const char* id)
      : internal_id(ObjectIDFromString(id)) {}
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

namespace detail {

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <py::return_value_policy Policy =
              py::return_value_policy::reference_internal,
          typename IteratorState, typename F, typename... Extra>
py::iterator make_iterator_fmap(IteratorState const& state, F functor,
                                Extra&&... extra) {
  if (!py::detail::get_type_info(typeid(IteratorState), false)) {
    py::class_<IteratorState>(py::handle(), "iterator",
                              pybind11::module_local())
        .def("__iter__", [](IteratorState& s) -> IteratorState& { return s; })
        .def(
            "__next__",
            [functor](IteratorState& s) -> py::object {
              if (!s.first_or_done)
                ++s.it;
              else
                s.first_or_done = false;
              if (s.it == s.end) {
                s.first_or_done = true;
                throw py::stop_iteration();
              }
              return functor(s.arg, s.it);
            },
            std::forward<Extra>(extra)..., Policy);
  }
  return py::cast(state);
}
}  // namespace detail
}  // namespace pybind11

namespace vineyard {

void throw_on_error(Status const& status);

/**
 * Copy a memoryview/buffer to a dst pointer.
 *
 * @size: capacity of the dst memory block.
 */
// dst[offset:offset+src.size()] = src[:]
// assert: dst.size() >= offset + src.size()
Status copy_memoryview(
    PyObject* dst, PyObject* src, size_t const offset = 0,
    size_t const concurrency = memory::default_memcpy_concurrency);

// dst[offset:offset+len(src)] = src[:]
// assert: dst_size >= offset + src.size()
Status copy_memoryview(
    void* dst, size_t const dst_size, PyObject* src, size_t const offset = 0,
    size_t const concurrency = memory::default_memcpy_concurrency);

// dst[offset:offset+src_size] = src[:]
// assert: dst.size() >= offset + src_size
Status copy_memoryview(
    PyObject* dst, const void* src, size_t const src_size,
    size_t const offset = 0,
    size_t const concurrency = memory::default_memcpy_concurrency);

// dst[offset:offset+src_size] = src[:]
// assert: dst_size >= offset + src_size
Status copy_memoryview(
    void* dst, size_t const dst_size, const void* src, size_t const src_size,
    size_t const offset = 0,
    size_t const concurrency = memory::default_memcpy_concurrency);

namespace detail {
py::object from_json(const json& value);
json to_json(const py::handle& obj);
}  // namespace detail

}  // namespace vineyard

#endif  // PYTHON_PYBIND11_UTILS_H_
