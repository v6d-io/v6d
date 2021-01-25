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

#include <functional>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#pragma GCC visibility push(default)
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/status.h"
#pragma GCC visibility pop

#include "pybind11_utils.h"  // NOLINT(build/include)

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces)

namespace vineyard {

void bind_core(py::module& mod) {
  // ObjectMeta
  py::class_<ObjectMeta>(mod, "ObjectMeta")
      .def(py::init<>())
      .def_property("__client", &ObjectMeta::GetClient, &ObjectMeta::SetClient)
      .def_property(
          "id",
          [](ObjectMeta* self) -> ObjectIDWrapper { return self->GetId(); },
          [](ObjectMeta* self, ObjectIDWrapper const id) { self->SetId(id); })
      .def_property_readonly("signature", &ObjectMeta::GetSignature)
      .def_property("typename", &ObjectMeta::GetTypeName,
                    &ObjectMeta::SetTypeName)
      .def_property("nbytes", &ObjectMeta::GetNBytes, &ObjectMeta::SetNBytes)
      .def_property_readonly("instance_id", &ObjectMeta::GetInstanceId)
      .def_property_readonly("islocal", &ObjectMeta::IsLocal)
      .def_property_readonly("isglobal", &ObjectMeta::IsGlobal)
      .def("set_global", [](ObjectMeta *self, const bool global) {
        self->SetGlobal(global);
      }, py::arg("global") = true)
      .def("__contains__", &ObjectMeta::Haskey, "key"_a)
      .def(
          "__getitem__",
          [](ObjectMeta* self, std::string const& key) -> py::object {
            auto const& tree = self->MetaData();
            auto iter = tree.find(key);
            if (iter == tree.end()) {
              throw py::key_error("key '" + key + "' does not exist");
            }
            if (!iter->is_object()) {
              return json_to_python(*iter);
            } else {
              return py::cast(self->GetMemberMeta(key));
            }
          },
          "key"_a)
      .def(
          "get",
          [](ObjectMeta* self, std::string const& key,
             py::object default_value) -> py::object {
            auto const& tree = self->MetaData();
            auto iter = tree.find(key);
            if (iter == tree.end()) {
              return default_value;
            }
            if (!iter->is_object()) {
              return json_to_python(*iter);
            } else {
              return py::cast(self->GetMemberMeta(key));
            }
          },
          "key"_a, py::arg("default") = py::none())
      .def("get_member",
           [](ObjectMeta* self, std::string const& key) -> py::object {
             auto const& tree = self->MetaData();
             auto iter = tree.find(key);
             if (iter == tree.end()) {
               return py::none();
             }
             VINEYARD_ASSERT(iter->is_object() && !iter->empty(),
                             "The value is not a member, but a meta");
             return py::cast(self->GetMember(key));
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::string const& value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             int32_t value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             int64_t value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             float value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             double value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             bool value) { self->AddKeyValue(key, value); })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::vector<int32_t> const& value) {
             self->AddKeyValue(key, value);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::vector<int64_t> const& value) {
             self->AddKeyValue(key, value);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::vector<float> const& value) {
             self->AddKeyValue(key, value);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::vector<double> const& value) {
             self->AddKeyValue(key, value);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::vector<std::string> const& value) {
             self->AddKeyValue(key, value);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key, Object const* member) {
             self->AddMember(key, member);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              ObjectIDWrapper const member) { self->AddMember(key, member); })
      .def("add_member",
           [](ObjectMeta* self, std::string const& key, Object const* member) {
             self->AddMember(key, member);
           })
      .def("add_member",
           [](ObjectMeta* self, std::string const& key,
              ObjectIDWrapper const member) { self->AddMember(key, member); })
      .def(
          "__iter__",
          [](const ObjectMeta& meta) {
            std::function<py::object(ObjectMeta::const_iterator &)> fn = [](
              ObjectMeta::const_iterator &iter) {
              return py::cast(iter.key());
            };
            return py::make_iterator_fmap(meta.begin(), meta.end(), fn);
          },
          py::keep_alive<0, 1>())
      .def(
          "items",
          [](const ObjectMeta& meta) {
            std::function<py::object(ObjectMeta::const_iterator &)> fn = [&meta](
              ObjectMeta::const_iterator &iter) {
              if (iter.value().is_object()) {
                return py::cast(meta.GetMemberMeta(iter.key()));
              } else {
                return json_to_python(iter.value());
              }
            };
            return py::make_iterator_fmap(meta.begin(), meta.end(), fn);
          },
          py::keep_alive<0, 1>())
      .def("__repr__",
           [](const ObjectMeta* meta) {
             thread_local std::stringstream ss;
             return meta->MetaData().dump(4);
           })
      .def("__str__", [](const ObjectMeta* meta) {
        thread_local std::stringstream ss;
        ss.str("");
        ss.clear();
        ss << "ObjectMeta ";
        ss << meta->MetaData().dump(4);
        return ss.str();
      });

  // ObjectIDWrapper
  py::class_<ObjectIDWrapper>(mod, "ObjectID")
      .def(py::init<>())
      .def(py::init<ObjectID>(), "id"_a)
      .def(py::init<std::string const&>(), "id"_a)
      .def("__int__", [](const ObjectIDWrapper& id) { return ObjectID(id); })
      .def("__repr__",
           [](const ObjectIDWrapper& id) { return VYObjectIDToString(id); })
      .def("__str__",
           [](const ObjectIDWrapper& id) {
             return "ObjectID <\"" + VYObjectIDToString(id) + "\">";
           })
      .def(py::pickle(
          [](const ObjectIDWrapper& id) {  // __getstate__
            return py::make_tuple(ObjectID(id));
          },
          [](py::tuple const& tup) {  // __setstate__
            if (tup.size() != 1) {
              throw std::runtime_error(
                  "Invalid state, cannot be pickled as ObjectID!");
            }
            return ObjectIDWrapper{tup[0].cast<ObjectID>()};
          }));

  // Object
  py::class_<Object, std::shared_ptr<Object>>(mod, "Object")
      .def_property_readonly(
          "id", [](Object* self) -> ObjectIDWrapper { return self->id(); })
      .def_property_readonly(
          "signature", [](Object* self) -> Signature { return self->meta().GetSignature(); })
      .def_property_readonly("meta", &Object::meta,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("nbytes", &Object::nbytes)
      .def_property_readonly("typename",
                             [](const Object* object) -> const std::string {
                               return object->meta().GetTypeName();
                             })
      .def("member",
           [](Object const* self, std::string const& name) {
             return self->meta().GetMember(name);
           })
      .def_property_readonly("islocal", &Object::IsLocal)
      .def_property_readonly("ispersist", &Object::IsPersist)
      .def_property_readonly("isglobal", &Object::IsGlobal)
      .def("__repr__",
           [](const Object* self) {
             return "Object <\"" + VYObjectIDToString(self->id()) +
                    "\": " + self->meta().GetTypeName() + ">";
           })
      .def("__str__", [](const Object* self) {
        return "Object <\"" + VYObjectIDToString(self->id()) +
               "\": " + self->meta().GetTypeName() + ">";
      });

  // ObjectBuilder
  py::class_<ObjectBuilder, std::shared_ptr<ObjectBuilder>>(mod,
                                                            "ObjectBuilder")
      // NB: don't expose the "Build" method to python.
      .def("seal", &ObjectBuilder::Seal, "client"_a)
      .def_property_readonly("issealed", &ObjectBuilder::sealed);

  // Blob
  py::class_<Blob, std::shared_ptr<Blob>, Object>(mod, "Blob",
                                                  py::buffer_protocol())
      .def_property_readonly("size", &Blob::size)
      .def_static("empty", &Blob::MakeEmpty)
      .def("__len__", &Blob::size)
      .def("__iter__",
           [](Blob* self) {
             auto data = self->data();
             auto size = self->size();
             return py::make_iterator(data, data + size);
           })
      .def(
          "__getitem__",
          [](BlobWriter* self, size_t const index) -> int8_t {
            // NB: unchecked
            return self->data()[index];
          },
          "index"_a)
      .def("__dealloc__", [](Blob* self) {})
      .def_property_readonly(
          "address",
          [](Blob* self) { return reinterpret_cast<uintptr_t>(self->data()); })
      .def_property_readonly(
          "buffer",
          [](Blob& blob) -> py::object {
            auto pa = py::module::import("pyarrow");
            auto buffer = blob.Buffer();
            if (buffer == nullptr) {
              return py::none();
            } else {
              return pa.attr("py_buffer")(py::memoryview::from_memory(
                  const_cast<uint8_t*>(buffer->data()), buffer->size(), true));
            }
          })
      .def_buffer([](Blob& blob) -> py::buffer_info {
        return py::buffer_info(const_cast<char*>(blob.data()), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, true);
      });

  // BlobBuilder
  py::class_<BlobWriter, std::shared_ptr<BlobWriter>, ObjectBuilder>(
      mod, "BlobBuilder", py::buffer_protocol())
      .def_property_readonly("size", &BlobWriter::size)
      .def("__len__", &BlobWriter::size)
      .def("__iter__",
           [](Blob* self) {
             auto data = self->data();
             auto size = self->size();
             return py::make_iterator(data, data + size);
           })
      .def(
          "__getitem__",
          [](BlobWriter* self, size_t const index) -> int8_t {
            // NB: unchecked
            return self->data()[index];
          },
          "index_a")
      .def(
          "__setitem__",
          [](BlobWriter* self, size_t const index, int8_t const value) {
            // NB: unchecked
            self->data()[index] = value;
          },
          "index"_a, "value"_a)
      .def(
          "__setitem__",
          [](BlobWriter* self, std::string const& key,
             std::string const& value) {
            // NB: unchecked
            self->AddKeyValue(key, value);
          },
          "key"_a, "value"_a)
      .def(
          "copy",
          [](BlobWriter* self, size_t const offset, uintptr_t ptr,
             size_t const size) {
            std::memcpy(self->data() + offset, reinterpret_cast<void*>(ptr),
                        size);
          },
          "offset"_a, "address"_a, "size"_a)
      .def(
          "copy",
          [](BlobWriter* self, size_t offset, py::bytes bs) {
            // FIXME: avoid this explicit copy
            std::string ss = static_cast<std::string>(bs);
            VINEYARD_ASSERT(offset + ss.size() <= self->size());
            std::memcpy(self->data() + offset, ss.c_str(), ss.size());
          },
          "offset"_a, "bytes"_a)
      .def_property_readonly("address",
                             [](BlobWriter* self) {
                               return reinterpret_cast<uintptr_t>(self->data());
                             })
      .def_property_readonly(
          "buffer",
          [](BlobWriter& blob) -> py::object {
            auto pa = py::module::import("pyarrow");
            auto buffer = blob.Buffer();
            if (buffer == nullptr) {
              return py::none();
            } else {
              return pa.attr("py_buffer")(py::memoryview::from_memory(
                  buffer->mutable_data(), buffer->size(), false));
            }
          })
      .def_buffer([](BlobWriter& blob) -> py::buffer_info {
        return py::buffer_info(blob.data(), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, false);
      });
}

}  // namespace vineyard
