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

#include "pybind11_utils.h"  // NOLINT(build/include)

#include <cstdlib>
#include <cstring>

#include "pybind11/pybind11.h"

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces)

namespace vineyard {

/**
 * Variant for :code:`pybind11::memoryview::memory`, but supports borrowed
 * memory.
 */
py::memoryview memoryview_from_memory(void* mem, ssize_t size, bool readonly,
                                      bool borrowed) {
  PyObject* ptr =
      PyMemoryView_FromMemory(reinterpret_cast<char*>(mem), size,
                              (readonly) ? PyBUF_READ : PyBUF_WRITE);
  if (!ptr) {
    py::pybind11_fail("Could not allocate memoryview object!");
  }
  return py::memoryview(py::object(ptr, borrowed));
}

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
    PyCFunctionObject* f = (PyCFunctionObject*) obj;
    if (f->m_ml->ml_doc && f->m_ml->ml_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "function '%s' already has a docstring",
                          f->m_ml->ml_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    f->m_ml->ml_doc = strndup(doc_str, doc_len);
  } else if (Py_TYPE(obj) == &PyInstanceMethod_Type) {
    PyCFunctionObject* f =
        (PyCFunctionObject*) (PyInstanceMethod_GET_FUNCTION(obj));
    if (f->m_ml->ml_doc && f->m_ml->ml_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "function '%s' already has a docstring",
                          f->m_ml->ml_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    f->m_ml->ml_doc = strndup(doc_str, doc_len);
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = (PyMethodDescrObject*) obj;
    if (m->d_method->ml_doc && m->d_method->ml_doc[0]) {
      return PyErr_Format(PyExc_RuntimeError,
                          "method '%s' already has a docstring",
                          m->d_method->ml_name);
    }
    // copy the doc string since pybind11 will release the doc unconditionally.
    m->d_method->ml_doc = strndup(doc_str, doc_len);
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    PyGetSetDescrObject* m = (PyGetSetDescrObject*) obj;
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
    PyTypeObject* t = (PyTypeObject*) obj;
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
  PyModule_AddFunctions(mod.ptr(), vineyard_utils_methods);
}

}  // namespace vineyard
