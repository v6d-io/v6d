/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_utils.h"  // NOLINT(build/include_subdir)

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces_literals)

namespace vineyard {

namespace detail {
/// Extends pybind11::detail::iterator_state to holds reference to a
/// stable (unchanged) globally available "argument".
template <typename Arg>
struct metadata_iterator_state {
  ObjectMeta::const_iterator it;
  ObjectMeta::const_iterator end;
  bool first_or_done;
  Arg arg;
};
}  // namespace detail

void bind_core(py::module& mod) {
  // ObjectIDWrapper
  py::class_<ObjectIDWrapper>(mod, "ObjectID")
      .def(py::init<>())
      .def(py::init<ObjectID>(), "id"_a)
      .def(py::init<std::string const&>(), "id"_a)
      .def("__int__", [](const ObjectIDWrapper& id) { return ObjectID(id); })
      .def("__repr__",
           [](const ObjectIDWrapper& id) { return ObjectIDToString(id); })
      .def("__str__",
           [](const ObjectIDWrapper& id) {
             return "ObjectID <\"" + ObjectIDToString(id) + "\">";
           })
      .def("__hash__", [](const ObjectIDWrapper& id) { return ObjectID(id); })
      .def("__eq__", [](const ObjectIDWrapper& id,
                        const ObjectIDWrapper& other) { return id == other; })
      .def_property_readonly("uri",
                             [](const ObjectIDWrapper& id) {
                               return "vineyard://" + ObjectIDToString(id);
                             })
      .def_static(
          "from_uri",
          [](const std::string& uri) -> py::object {
            if (uri.find("vineyard://") == 0) {
              return py::cast(
                  ObjectIDWrapper(ObjectIDFromString(uri.substr(11))));
            } else {
              throw_on_error(Status::UserInputError(
                  "Not a valid uri for vineyard object ID"));
              return py::none();
            }
          },
          "uri"_a)
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

  // ObjectNameWrapper
  py::class_<ObjectNameWrapper>(mod, "ObjectName")
      .def(py::init<std::string const&>(), "name"_a)
      .def("__repr__",
           [](const ObjectNameWrapper& name) {
             return py::repr(py::cast(std::string(name)));
           })
      .def("__str__",
           [](const ObjectNameWrapper& name) {
             return py::str(py::cast(std::string(name)));
           })
      .def("__hash__",
           [](const ObjectNameWrapper& name) {
             return py::hash(py::cast(std::string(name)));
           })
      .def("__eq__",
           [](const ObjectNameWrapper& name, const ObjectNameWrapper& other) {
             return name == other;
           })
      .def_property_readonly("uri",
                             [](const ObjectNameWrapper& name) {
                               return "vineyard://" + std::string(name);
                             })
      .def_static(
          "from_uri",
          [](const std::string& uri) -> py::object {
            if (uri.find("vineyard://") == 0) {
              return py::cast(ObjectNameWrapper(uri.substr(11)));
            } else {
              throw_on_error(Status::UserInputError(
                  "Not a valid uri for vineyard object name"));
              return py::none();
            }
          },
          "uri"_a)
      .def(py::pickle(
          [](const ObjectNameWrapper& name) {  // __getstate__
            return py::make_tuple(py::cast(std::string(name)));
          },
          [](py::tuple const& tup) {  // __setstate__
            if (tup.size() != 1) {
              throw std::runtime_error(
                  "Invalid state, cannot be pickled as ObjectName!");
            }
            return ObjectNameWrapper{tup[0].cast<std::string>()};
          }));

  // ObjectMeta
  py::class_<ObjectMeta>(mod, "ObjectMeta")
      .def(py::init<>([](bool global_, py::args,
                         py::kwargs) -> std::unique_ptr<ObjectMeta> {
             std::unique_ptr<ObjectMeta> meta(new ObjectMeta());
             meta->SetGlobal(global_);
             return meta;
           }),
           py::arg("global_") = false)
      .def_property("_client", &ObjectMeta::GetClient, &ObjectMeta::SetClient)
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
      .def(
          "set_global",
          [](ObjectMeta* self, const bool global) { self->SetGlobal(global); },
          py::arg("global") = true)
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
              return detail::from_json(*iter);
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
              return detail::from_json(*iter);
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
      .def("get_buffer",
           [](ObjectMeta* self, const ObjectID key) -> py::memoryview {
             std::shared_ptr<arrow::Buffer> buffer;
             throw_on_error(self->GetBuffer(key, buffer));
             if (buffer == nullptr) {
               return py::none();
             } else {
               return py::memoryview::from_memory(
                   const_cast<uint8_t*>(buffer->data()), buffer->size(), true);
             }
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key) {
             self->AddKeyValueFromEnv(key);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              std::string const& value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             bool value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             int32_t value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             int64_t value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             float value) { self->AddKeyValue(key, value); })
      .def("__setitem__", [](ObjectMeta* self, std::string const& key,
                             double value) { self->AddKeyValue(key, value); })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key, Object const* member) {
             self->AddMember(key, member);
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              ObjectMeta const& member) { self->AddMember(key, member); })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key,
              ObjectIDWrapper const member) { self->AddMember(key, member); })
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
           [](ObjectMeta* self, std::string const& key, py::list const& value) {
             self->AddKeyValue(key, detail::to_json(value));
           })
      .def("__setitem__",
           [](ObjectMeta* self, std::string const& key, py::dict const& value) {
             self->AddKeyValue(key, detail::to_json(value));
           })
      .def("add_member",
           [](ObjectMeta* self, std::string const& key, Object const* member) {
             self->AddMember(key, member);
           })
      .def("add_member",
           [](ObjectMeta* self, std::string const& key,
              ObjectMeta const& member) { self->AddMember(key, member); })
      .def("add_member",
           [](ObjectMeta* self, std::string const& key,
              ObjectIDWrapper const member) { self->AddMember(key, member); })
      .def("reset", [](ObjectMeta& meta) { meta.Reset(); })
      .def_property_readonly("memory_usage", &ObjectMeta::MemoryUsage)
      .def("reset_key",
           [](ObjectMeta& meta, std::string const& key) { meta.ResetKey(key); })
      .def("reset_signature", [](ObjectMeta& meta) { meta.ResetSignature(); })
      .def(
          "__iter__",
          [](const ObjectMeta& meta) {
            std::function<py::object(std::true_type,
                                     ObjectMeta::const_iterator&)>
                fn = [](std::true_type, ObjectMeta::const_iterator& iter) {
                  return py::cast(iter.key());
                };
            using state_t = detail::metadata_iterator_state<std::true_type>;
            return py::detail::make_iterator_fmap(
                state_t{meta.begin(), meta.end(), true, std::true_type{}}, fn);
          },
          py::keep_alive<0, 1>())
      .def(
          "items",
          [](const ObjectMeta& meta) {
            std::function<py::object(const ObjectMeta&,
                                     ObjectMeta::const_iterator&)>
                fn = [](const ObjectMeta& meta,
                        ObjectMeta::const_iterator& iter) -> py::object {
              if (iter.value().is_object()) {
                return py::cast(std::make_pair(
                    iter.key(), py::cast(meta.GetMemberMeta(iter.key()))));
              } else {
                return py::cast(std::make_pair(
                    iter.key(), detail::from_json(iter.value())));
              }
            };
            using state_t = detail::metadata_iterator_state<const ObjectMeta&>;
            return py::detail::make_iterator_fmap(
                state_t{meta.begin(), meta.end(), true, std::cref(meta)}, fn);
          },
          py::keep_alive<0, 1>())
      .def("__repr__",
           [](const ObjectMeta* meta) {
             thread_local std::stringstream ss;
             return meta->MetaData().dump(4);
           })
      .def("__str__",
           [](const ObjectMeta* meta) {
             thread_local std::stringstream ss;
             ss.str("");
             ss.clear();
             ss << "ObjectMeta ";
             ss << meta->MetaData().dump(4);
             return ss.str();
           })
      .def(py::pickle(
          [](const ObjectMeta& meta) {  // __getstate__
            return py::make_tuple(detail::from_json(meta.MetaData()));
          },
          [](py::tuple const& tup) -> ObjectMeta {  // __setstate__
            if (tup.size() != 1) {
              throw std::runtime_error(
                  "Invalid state, cannot be pickled as ObjectID!");
            }
            ObjectMeta meta;
            meta.SetMetaData(nullptr, detail::to_json(tup[0]));
            return meta;
          }));

  // Object
  py::class_<Object, std::shared_ptr<Object>>(mod, "Object")
      .def_property_readonly(
          "id", [](Object* self) -> ObjectIDWrapper { return self->id(); })
      .def_property_readonly(
          "signature",
          [](Object* self) -> Signature { return self->meta().GetSignature(); })
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
      .def("__getitem__",
           [](Object const* self, std::string const& name) {
             return self->meta().GetMember(name);
           })
      .def_property_readonly("islocal", &Object::IsLocal)
      .def_property_readonly("ispersist", &Object::IsPersist)
      .def_property_readonly("isglobal", &Object::IsGlobal)
      .def("__repr__",
           [](const Object* self) {
             return "Object <\"" + ObjectIDToString(self->id()) +
                    "\": " + self->meta().GetTypeName() + ">";
           })
      .def("__str__", [](const Object* self) {
        return "Object <\"" + ObjectIDToString(self->id()) +
               "\": " + self->meta().GetTypeName() + ">";
      });

  // ObjectBuilder
  py::class_<ObjectBuilder, std::shared_ptr<ObjectBuilder>>(mod,
                                                            "ObjectBuilder")
      // NB: don't expose the "Build" method to python.
      .def("seal", &ObjectBuilder::Seal, "client"_a)
      .def_property_readonly("issealed", &ObjectBuilder::sealed);
}

void bind_blobs(py::module& mod) {
  // Blob
  py::class_<Blob, std::shared_ptr<Blob>, Object>(mod, "Blob",
                                                  py::buffer_protocol())
      .def_property_readonly("size", &Blob::size)
      .def_property_readonly(
          "is_empty", [](Blob* self) { return self->allocated_size() == 0; })
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
          [](Blob* self, size_t const index) -> int8_t {
            // NB: unchecked
            return self->data()[index];
          },
          "index"_a)
      .def("__dealloc__", [](Blob* self) {})
      .def_property_readonly(
          "address",
          [](Blob* self) { return reinterpret_cast<uintptr_t>(self->data()); })
      .def_property_readonly("buffer",
                             [](Blob& blob) -> py::object {
                               auto buffer = blob.Buffer();
                               if (buffer == nullptr) {
                                 return py::none();
                               } else {
                                 return py::memoryview::from_memory(
                                     const_cast<uint8_t*>(buffer->data()),
                                     buffer->size(), true);
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
      .def_property_readonly(
          "id", [](BlobWriter* self) -> ObjectIDWrapper { return self->id(); })
      .def_property_readonly("size", &BlobWriter::size)
      .def("__len__", &BlobWriter::size)
      .def("__iter__",
           [](BlobWriter* self) {
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
          "abort",
          [](BlobWriter* self, Client& client) {
            throw_on_error(self->Abort(client));
          },
          "client"_a)
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
          [](BlobWriter* self, size_t offset, py::buffer const& buffer) {
            throw_on_error(copy_memoryview(buffer.ptr(), self->data(),
                                           self->size(), offset));
          },
          "offset"_a, "buffer"_a)
      .def(
          "copy",
          [](BlobWriter* self, size_t offset, py::bytes const& bs) {
            char* buffer = nullptr;
            ssize_t length = 0;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bs.ptr(), &buffer, &length)) {
              py::pybind11_fail("Unable to extract bytes contents!");
            }
            if (offset + length > self->size()) {
              throw_on_error(Status::AssertionFailed(
                  "Expect a source buffer with size at most '" +
                  std::to_string(self->size() - offset) +
                  "', but the buffer size is '" + std::to_string(length) +
                  "'"));
            }
            std::memcpy(self->data() + offset, buffer, length);
          },
          "offset"_a, "bytes"_a)
      .def_property_readonly("address",
                             [](BlobWriter* self) {
                               return reinterpret_cast<uintptr_t>(self->data());
                             })
      .def_property_readonly("buffer",
                             [](BlobWriter& blob) -> py::object {
                               auto buffer = blob.Buffer();
                               if (buffer == nullptr) {
                                 return py::none();
                               } else {
                                 return py::memoryview::from_memory(
                                     buffer->mutable_data(), buffer->size(),
                                     false);
                               }
                             })
      .def_buffer([](BlobWriter& blob) -> py::buffer_info {
        return py::buffer_info(blob.data(), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, false);
      });

  // RemoteBlob
  py::class_<RemoteBlob, std::shared_ptr<RemoteBlob>>(mod, "RemoteBlob",
                                                      py::buffer_protocol())
      .def_property_readonly(
          "id", [](RemoteBlob* self) -> ObjectIDWrapper { return self->id(); })
      .def_property_readonly(
          "instance_id",
          [](RemoteBlob* self) -> InstanceID { return self->instance_id(); })
      .def_property_readonly("size", &RemoteBlob::size)
      .def_property_readonly("allocated_size", &RemoteBlob::allocated_size)
      .def_property_readonly(
          "is_empty",
          [](RemoteBlob* self) { return self->allocated_size() == 0; })
      .def("__len__", &RemoteBlob::allocated_size)
      .def("__iter__",
           [](RemoteBlob* self) {
             auto data = self->data();
             auto size = self->size();
             return py::make_iterator(data, data + size);
           })
      .def(
          "__getitem__",
          [](RemoteBlob* self, size_t const index) -> int8_t {
            // NB: unchecked
            return self->data()[index];
          },
          "index"_a)
      .def("__dealloc__", [](RemoteBlob* self) {})
      .def_property_readonly("address",
                             [](RemoteBlob* self) {
                               return reinterpret_cast<uintptr_t>(self->data());
                             })
      .def_property_readonly("buffer",
                             [](RemoteBlob& blob) -> py::object {
                               auto buffer = blob.Buffer();
                               if (buffer == nullptr) {
                                 return py::none();
                               } else {
                                 return py::memoryview::from_memory(
                                     const_cast<uint8_t*>(buffer->data()),
                                     buffer->size(), true);
                               }
                             })
      .def_buffer([](RemoteBlob& blob) -> py::buffer_info {
        return py::buffer_info(const_cast<char*>(blob.data()), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, true);
      });

  // RemoteBlobBuilder
  py::class_<RemoteBlobWriter, std::shared_ptr<RemoteBlobWriter>>(
      mod, "RemoteBlobBuilder", py::buffer_protocol())
      .def(py::init<>(
               [](const size_t size) -> std::unique_ptr<RemoteBlobWriter> {
                 return std::make_unique<RemoteBlobWriter>(size);
               }),
           py::arg("size"))
      .def_property_readonly("size", &RemoteBlobWriter::size)
      .def("__len__", &RemoteBlobWriter::size)
      .def("__iter__",
           [](RemoteBlobWriter* self) {
             auto data = self->data();
             auto size = self->size();
             return py::make_iterator(data, data + size);
           })
      .def(
          "__getitem__",
          [](RemoteBlobWriter* self, size_t const index) -> int8_t {
            // NB: unchecked
            return self->data()[index];
          },
          "index_a")
      .def(
          "__setitem__",
          [](RemoteBlobWriter* self, size_t const index, int8_t const value) {
            // NB: unchecked
            self->data()[index] = value;
          },
          "index"_a, "value"_a)
      .def(
          "abort",
          [](RemoteBlobWriter* self, Client& client) {
            throw_on_error(self->Abort());
          },
          "client"_a)
      .def(
          "copy",
          [](RemoteBlobWriter* self, size_t const offset, uintptr_t ptr,
             size_t const size) {
            std::memcpy(self->data() + offset, reinterpret_cast<void*>(ptr),
                        size);
          },
          "offset"_a, "address"_a, "size"_a)
      .def(
          "copy",
          [](RemoteBlobWriter* self, size_t offset, py::buffer const& buffer) {
            throw_on_error(copy_memoryview(buffer.ptr(), self->data(),
                                           self->size(), offset));
          },
          "offset"_a, "buffer"_a)
      .def(
          "copy",
          [](RemoteBlobWriter* self, size_t offset, py::bytes const& bs) {
            char* buffer = nullptr;
            ssize_t length = 0;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bs.ptr(), &buffer, &length)) {
              py::pybind11_fail("Unable to extract bytes contents!");
            }
            if (offset + length > self->size()) {
              throw_on_error(Status::AssertionFailed(
                  "Expect a source buffer with size at most '" +
                  std::to_string(self->size() - offset) +
                  "', but the buffer size is '" + std::to_string(length) +
                  "'"));
            }
            std::memcpy(self->data() + offset, buffer, length);
          },
          "offset"_a, "bytes"_a)
      .def_property_readonly("address",
                             [](RemoteBlobWriter* self) {
                               return reinterpret_cast<uintptr_t>(self->data());
                             })
      .def_property_readonly("buffer",
                             [](RemoteBlobWriter& blob) -> py::object {
                               auto buffer = blob.Buffer();
                               if (buffer == nullptr) {
                                 return py::none();
                               } else {
                                 return py::memoryview::from_memory(
                                     buffer->mutable_data(), buffer->size(),
                                     false);
                               }
                             })
      .def_buffer([](RemoteBlobWriter& blob) -> py::buffer_info {
        return py::buffer_info(blob.data(), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, false);
      });
}

}  // namespace vineyard
