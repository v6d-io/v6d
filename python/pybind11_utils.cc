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

#include "pybind11_utils.h"  // NOLINT(build/include_subdir)

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "common/memory/memcpy.h"
#include "common/util/json.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces_literals)

namespace vineyard {

static PyObject* vineyard_add_doc(PyObject* /* unused */, PyObject* args) {
  PyObject* obj;
  PyObject* doc_obj;
  if (!PyArg_ParseTuple(args, "OO", &obj, &doc_obj)) {
    return PyErr_Format(PyExc_RuntimeError,
                        "Two arguments (value, doc) is required");
  }

  // adds a __doc__ string to a function, similar to pytorch's _add_docstr
  const char* doc_str = "<invalid string>";
  Py_ssize_t doc_len = std::strlen(doc_str);
  if (PyBytes_Check(doc_obj)) {
    doc_len = PyBytes_GET_SIZE(doc_obj);
    doc_str = PyBytes_AS_STRING(doc_obj);
  }
  if (PyUnicode_Check(doc_obj)) {
    doc_str = PyUnicode_AsUTF8AndSize(doc_obj, &doc_len);
    if (!doc_str) {
      return PyErr_Format(PyExc_RuntimeError,
                          "error unpacking string as utf-8");
    }
  }

  if (Py_TYPE(obj) == &PyCFunction_Type) {
    PyCFunctionObject* f =
        (PyCFunctionObject*) obj;  // NOLINT(readability/casting)
    if (f->m_ml->ml_doc && f->m_ml->ml_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "function '%s' already has a docstring",
                          f->m_ml->ml_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    f->m_ml->ml_doc = strndup(doc_str, doc_len);
  } else if (Py_TYPE(obj) == &PyInstanceMethod_Type) {
    PyObject* fobj = PyInstanceMethod_GET_FUNCTION(obj);
    PyCFunctionObject* f =
        (PyCFunctionObject*) fobj;  // NOLINT(readability/casting)
    if (f->m_ml->ml_doc && f->m_ml->ml_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "function '%s' already has a docstring",
                          f->m_ml->ml_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    f->m_ml->ml_doc = strndup(doc_str, doc_len);
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m =
        (PyMethodDescrObject*) obj;  // NOLINT(readability/casting)
    if (m->d_method->ml_doc && m->d_method->ml_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "method '%s' already has a docstring",
                          m->d_method->ml_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    m->d_method->ml_doc = strndup(doc_str, doc_len);
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    PyGetSetDescrObject* m =
        (PyGetSetDescrObject*) obj;  // NOLINT(readability/casting)
    if (m->d_getset->doc && m->d_getset->doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "attribute '%s' already has a docstring",
                          m->d_getset->name);
    }
    // This field is not const for python < 3.7 yet the content is
    // never modified.
    // copy the doc string since pybind11 will release the doc unconditionally.
    m->d_getset->doc = const_cast<char*>(strndup(doc_str, doc_len));
  } else if (Py_TYPE(obj) == &PyType_Type) {
    PyTypeObject* t = (PyTypeObject*) obj;  // NOLINT(readability/casting)
    if (t->tp_doc && t->tp_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "Type '%s' already has a docstring", t->tp_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    t->tp_doc = strndup(doc_str, doc_len);
  } else {
    PyObject* doc_attr = nullptr;
    doc_attr = PyObject_GetAttrString(obj, "__doc__");
    if (doc_attr != NULL && doc_attr != Py_None &&
        ((PyBytes_Check(doc_attr) && PyBytes_GET_SIZE(doc_attr) > 0) ||
         (PyUnicode_Check(doc_attr) && PyUnicode_GET_LENGTH(doc_attr) > 0))) {
      return PyErr_Format(PyExc_RuntimeError, "Object already has a docstring");
    }
    Py_XDECREF(doc_attr);
    Py_XINCREF(doc_obj);
    if (PyObject_SetAttrString(obj, "__doc__", doc_obj) < 0) {
      return PyErr_Format(PyExc_TypeError,
                          "Cannot set a docstring for that object");
    }
  }
  Py_RETURN_NONE;
}

static PyMethodDef vineyard_utils_methods[] = {
    {"_add_doc", vineyard_add_doc, METH_VARARGS,
     "Associate docstring with pybind11 exposed functions, types and methods."},
    {NULL, NULL, 0, NULL},
};

void bind_utils(py::module& mod) {
  mod.def(
      "memory_copy",
      [](py::buffer const dst, py::buffer const src, size_t offset = 0,
         size_t const concurrency = memory::default_memcpy_concurrency) {
        throw_on_error(
            copy_memoryview(dst.ptr(), src.ptr(), offset, concurrency));
      },
      "dst"_a, "src"_a, py::arg("offset") = 0,
      py::arg("concurrency") = memory::default_memcpy_concurrency);
  mod.def(
      "memory_copy",
      [](uintptr_t dst, size_t dst_size, py::buffer const src,
         size_t offset = 0,
         size_t const concurrency = memory::default_memcpy_concurrency) {
        throw_on_error(copy_memoryview(reinterpret_cast<void*>(dst), dst_size,
                                       src.ptr(), offset, concurrency));
      },
      "dst"_a, "dst_size"_a, "src"_a, py::arg("offset") = 0,
      py::arg("concurrency") = memory::default_memcpy_concurrency);
  mod.def(
      "memory_copy",
      [](py::buffer const dst, uintptr_t src, size_t src_size,
         size_t offset = 0,
         size_t const concurrency = memory::default_memcpy_concurrency) {
        throw_on_error(copy_memoryview(dst.ptr(), reinterpret_cast<void*>(src),
                                       src_size, offset, concurrency));
      },
      "dst"_a, "src"_a, "src_size"_a, py::arg("offset") = 0,
      py::arg("concurrency") = memory::default_memcpy_concurrency);
  mod.def(
      "memory_copy",
      [](uintptr_t dst, size_t dst_size, uintptr_t src, size_t src_size,
         size_t offset = 0,
         size_t const concurrency = memory::default_memcpy_concurrency) {
        throw_on_error(copy_memoryview(reinterpret_cast<void*>(dst), dst_size,
                                       reinterpret_cast<void*>(src), src_size,
                                       offset, concurrency));
      },
      "dst"_a, "dst_size"_a, "src"_a, "src_size"_a, py::arg("offset") = 0,
      py::arg("concurrency") = memory::default_memcpy_concurrency);

  PyModule_AddFunctions(mod.ptr(), vineyard_utils_methods);
}

namespace detail {

/**
 * Intend to be used as a manager to PyBuffer that releases the buffer
 * automatically when been destructed.
 */
class PyBufferGetter {
 public:
  explicit PyBufferGetter(PyObject* object) {
    has_buffer_ =
        PyObject_GetBuffer(object, &buffer_, PyBUF_ANY_CONTIGUOUS) == 0;
  }

  ~PyBufferGetter() {
    if (has_buffer_) {
      PyBuffer_Release(&buffer_);
    }
  }

  bool has_buffer() const { return has_buffer_; }

  uint8_t* data() const { return reinterpret_cast<uint8_t*>(buffer_.buf); }

  Py_ssize_t size() const { return buffer_.len; }

  bool readonly() const { return buffer_.readonly; }

 private:
  Py_buffer buffer_;
  bool has_buffer_;
};

}  // namespace detail

// dst[offset:offset+src.size()] = src[:]
// assert: dst.size() >= offset + src.size()
Status copy_memoryview(PyObject* dst, PyObject* src, size_t const offset,
                       size_t const concurrency) {
  detail::PyBufferGetter src_buffer(src);
  if (!src_buffer.has_buffer()) {
    return Status::AssertionFailed(
        "Not a contiguous memoryview for src, please consider translate it "
        "to `bytes` first.");
  }
  // skip none buffers
  if (src_buffer.data() == nullptr) {
    return Status::OK();
  }
  size_t src_size = src_buffer.size();

  detail::PyBufferGetter dst_buffer(dst);
  if (!dst_buffer.has_buffer()) {
    return Status::AssertionFailed(
        "Not a contiguous memoryview for dst, please consider translate it "
        "to `bytes` first.");
  }
  // skip none buffers
  if (dst_buffer.data() == nullptr) {
    return Status::OK();
  }
  size_t dst_size = dst_buffer.size();

  // validate expected size first
  if ((src_size != 0) && (src_size + offset > dst_size)) {
    return Status::AssertionFailed("Expect a source buffer with size at most'" +
                                   std::to_string(dst_size - offset) +
                                   "', but the buffer size is '" +
                                   std::to_string(src_size) + "'");
  }

  {
    py::gil_scoped_release release;
    // memcpy
    memory::concurrent_memcpy(
        reinterpret_cast<uint8_t*>(dst_buffer.data()) + offset,
        src_buffer.data(), src_size, concurrency);
  }
  return Status::OK();
}

// dst[offset:offset+len(src)] = src[:]
// assert: dst_size >= offset + src.size()
Status copy_memoryview(void* dst, size_t const dst_size, PyObject* src,
                       size_t const offset, size_t const concurrency) {
  detail::PyBufferGetter src_buffer(src);
  if (!src_buffer.has_buffer()) {
    return Status::AssertionFailed(
        "Not a contiguous memoryview for src, please consider translate it "
        "to `bytes` first.");
  }
  // skip none buffers
  if (src_buffer.data() == nullptr) {
    return Status::OK();
  }
  size_t src_size = src_buffer.size();

  // validate expected size first
  if ((src_size != 0) && (src_size + offset > dst_size)) {
    return Status::AssertionFailed("Expect a source buffer with size at most'" +
                                   std::to_string(dst_size - offset) +
                                   "', but the buffer size is '" +
                                   std::to_string(src_size) + "'");
  }

  {
    py::gil_scoped_release release;
    // memcpy
    memory::concurrent_memcpy(reinterpret_cast<uint8_t*>(dst) + offset,
                              src_buffer.data(), src_size, concurrency);
  }
  return Status::OK();
}

// dst[offset:offset+src_size] = src[:]
// assert: dst.size() >= offset + src_size
Status copy_memoryview(PyObject* dst, const void* src, size_t const src_size,
                       size_t const offset, size_t const concurrency) {
  detail::PyBufferGetter dst_buffer(dst);
  if (!dst_buffer.has_buffer()) {
    return Status::AssertionFailed(
        "Not a contiguous memoryview for dst, please consider translate it "
        "to `bytes` first.");
  }
  // skip none buffers
  if (dst_buffer.data() == nullptr) {
    return Status::OK();
  }
  size_t dst_size = dst_buffer.size();

  // validate expected size first
  if ((src_size != 0) && (src_size + offset > dst_size)) {
    return Status::AssertionFailed("Expect a source buffer with size at most'" +
                                   std::to_string(dst_size - offset) +
                                   "', but the buffer size is '" +
                                   std::to_string(src_size) + "'");
  }

  {
    py::gil_scoped_release release;
    // memcpy
    memory::concurrent_memcpy(
        reinterpret_cast<uint8_t*>(dst_buffer.data()) + offset, src, src_size,
        concurrency);
  }
  return Status::OK();
}

// dst[offset:offset+src_size] = src[:]
// assert: dst_size >= offset + src_size
Status copy_memoryview(void* dst, size_t const dst_size, const void* src,
                       size_t const src_size, size_t const offset,
                       size_t const concurrency) {
  // validate expected size first
  if ((src_size != 0) && (src_size + offset > dst_size)) {
    return Status::AssertionFailed("Expect a source buffer with size at most'" +
                                   std::to_string(dst_size - offset) +
                                   "', but the buffer size is '" +
                                   std::to_string(src_size) + "'");
  }

  {
    py::gil_scoped_release release;
    // memcpy
    memory::concurrent_memcpy(reinterpret_cast<uint8_t*>(dst) + offset, src,
                              src_size, concurrency);
  }
  return Status::OK();
}

namespace detail {

/**
 * Cast nlohmann::json to python object.
 *
 * Refer to https://github.com/pybind/pybind11_json
 */
py::object from_json(const json& j) {
  if (j.is_null()) {
    return py::none();
  } else if (j.is_boolean()) {
    return py::bool_(j.get<bool>());
  } else if (j.is_number_integer()) {
    return py::int_(j.get<json::number_integer_t>());
  } else if (j.is_number_unsigned()) {
    return py::int_(j.get<json::number_unsigned_t>());
  } else if (j.is_number_float()) {
    return py::float_(j.get<double>());
  } else if (j.is_string()) {
    return py::str(j.get<std::string>());
  } else if (j.is_array()) {
    py::list obj;
    for (const auto& el : j) {
      obj.append(from_json(el));
    }
    return std::move(obj);
  } else {
    // Object
    py::dict obj;
    for (json::const_iterator it = j.cbegin(); it != j.cend(); ++it) {
      obj[py::str(it.key())] = from_json(it.value());
    }
    return std::move(obj);
  }
}

/**
 * Cast python object to nlohmann::json.
 *
 * Refer to https://github.com/pybind/pybind11_json
 */
json to_json(const py::handle& obj) {
  if (obj.ptr() == nullptr || obj.is_none()) {
    return nullptr;
  }
  if (py::isinstance<py::bool_>(obj)) {
    return obj.cast<bool>();
  }
  if (py::isinstance<py::int_>(obj)) {
    try {
      json::number_integer_t s = obj.cast<json::number_integer_t>();
      if (py::int_(s).equal(obj)) {
        return s;
      }
    } catch (...) {}
    try {
      json::number_unsigned_t u = obj.cast<json::number_unsigned_t>();
      if (py::int_(u).equal(obj)) {
        return u;
      }
    } catch (...) {}
    throw std::runtime_error(
        "to_json received an integer out of range for both "
        "json::number_integer_t and json::number_unsigned_t type: " +
        py::repr(obj).cast<std::string>());
  }
  if (py::isinstance<py::float_>(obj)) {
    return obj.cast<double>();
  }
  if (py::isinstance<py::bytes>(obj)) {
    py::module base64 = py::module::import("base64");
    return base64.attr("b64encode")(obj)
        .attr("decode")("utf-8")
        .cast<std::string>();
  }
  if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto out = json::array();
    for (const py::handle value : obj) {
      out.push_back(to_json(value));
    }
    return out;
  }
  if (py::isinstance<py::dict>(obj)) {
    auto out = json::object();
    for (const py::handle key : obj) {
      out[py::str(key).cast<std::string>()] = to_json(obj[key]);
    }
    return out;
  }
  throw std::runtime_error("to_json not implemented for this type of object: " +
                           py::repr(obj).cast<std::string>());
}

}  // namespace detail

}  // namespace vineyard
