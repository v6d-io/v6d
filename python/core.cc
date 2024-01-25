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

#include <functional>
#include <memory>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"
#include "common/memory/memcpy.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_docs.h"   // NOLINT(build/include_subdir)
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
  py::class_<ObjectIDWrapper>(mod, "ObjectID", doc::ObjectID)
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
  py::class_<ObjectNameWrapper>(mod, "ObjectName", doc::ObjectName)
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
  py::class_<ObjectMeta>(mod, "ObjectMeta", doc::ObjectMeta)
      .def(py::init<>([](bool global_, py::args,
                         py::kwargs) -> std::unique_ptr<ObjectMeta> {
             std::unique_ptr<ObjectMeta> meta(new ObjectMeta());
             meta->SetGlobal(global_);
             return meta;
           }),
           py::arg("global_") = false, doc::ObjectMeta__init__)
      .def_property("_client", &ObjectMeta::GetClient, &ObjectMeta::SetClient)
      .def_property(
          "id",
          [](ObjectMeta* self) -> ObjectIDWrapper { return self->GetId(); },
          [](ObjectMeta* self, ObjectIDWrapper const id) { self->SetId(id); },
          doc::ObjectMeta_id)
      .def_property_readonly("signature", &ObjectMeta::GetSignature,
                             doc::ObjectMeta_signature)
      .def_property("typename", &ObjectMeta::GetTypeName,
                    &ObjectMeta::SetTypeName, doc::ObjectMeta_typename)
      .def_property("nbytes", &ObjectMeta::GetNBytes, &ObjectMeta::SetNBytes,
                    doc::ObjectMeta_nbyte)
      .def_property_readonly("instance_id", &ObjectMeta::GetInstanceId,
                             doc::ObjectMeta_instance_id)
      .def_property_readonly("islocal", &ObjectMeta::IsLocal,
                             doc::ObjectMeta_islocal)
      .def_property_readonly("isglobal", &ObjectMeta::IsGlobal,
                             doc::ObjectMeta_isglobal)
      .def_property_readonly("meta",
                             [](ObjectMeta* self) -> ObjectMeta* {
                               // as an alias to unify some APIs on `Object` and
                               // `ObjectMeta`.
                               return self;
                             })
      .def(
          "set_global",
          [](ObjectMeta* self, const bool global) { self->SetGlobal(global); },
          py::arg("global_") = true, doc::ObjectMeta_set_global)
      .def("__contains__", &ObjectMeta::HasKey, "key"_a,
           doc::ObjectMeta__contains__)
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
          "key"_a, doc::ObjectMeta__getitem__)
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
          "key"_a, py::arg("default") = py::none(), doc::ObjectMeta_get)
      .def(
          "get_member",
          [](ObjectMeta* self, std::string const& key) -> py::object {
            auto const& tree = self->MetaData();
            auto iter = tree.find(key);
            if (iter == tree.end()) {
              return py::none();
            }
            VINEYARD_ASSERT(iter->is_object() && !iter->empty(),
                            "The value is not a member, but a meta");
            return py::cast(self->GetMember(key));
          },
          doc::ObjectMeta_get_member)
      .def(
          "member" /* alias for get_member() */,
          [](ObjectMeta* self, std::string const& key) -> py::object {
            auto const& tree = self->MetaData();
            auto iter = tree.find(key);
            if (iter == tree.end()) {
              return py::none();
            }
            VINEYARD_ASSERT(iter->is_object() && !iter->empty(),
                            "The value is not a member, but a meta");
            return py::cast(self->GetMember(key));
          },
          doc::ObjectMeta_get_member)
      .def("get_buffer",
           [](ObjectMeta* self, const ObjectID key) -> py::memoryview {
             std::shared_ptr<Buffer> buffer;
             throw_on_error(self->GetBuffer(key, buffer));
             if (buffer == nullptr) {
               return py::none();
             } else {
               return py::memoryview::from_memory(
                   const_cast<uint8_t*>(buffer->data()), buffer->size(), true);
             }
           })
      .def(
          "__setitem__",
          [](ObjectMeta* self, std::string const& key,
             std::string const& value) { self->AddKeyValue(key, value); },
          doc::ObjectMeta__setitem__)
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
      .def(
          "add_member",
          [](ObjectMeta* self, std::string const& key, Object const* member) {
            self->AddMember(key, member);
          },
          doc::ObjectMeta_add_member)
      .def("add_member",
           [](ObjectMeta* self, std::string const& key,
              ObjectMeta const& member) { self->AddMember(key, member); })
      .def("add_member",
           [](ObjectMeta* self, std::string const& key,
              ObjectIDWrapper const member) { self->AddMember(key, member); })
      .def("reset", [](ObjectMeta& meta) { meta.Reset(); })
      .def_property_readonly(
          "memory_usage",
          [](ObjectMeta& meta) -> size_t { return meta.MemoryUsage(); },
          doc::ObjectMeta_memory_usage)
      .def(
          "memory_usage_details",
          [](ObjectMeta& meta, const bool pretty) -> py::object {
            json usages;
            meta.MemoryUsage(usages, pretty);
            return detail::from_json(usages);
          },
          py::arg("pretty") = true)
      .def_property_readonly("timestamp", &ObjectMeta::Timestamp)
      .def_property_readonly("labels",
                             [](const ObjectMeta* self) -> py::object {
                               return detail::from_json(self->Labels());
                             })
      .def(
          "label",
          [](const ObjectMeta* self, std::string const& key) -> std::string {
            return self->Label(key);
          },
          "key"_a)
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
  py::class_<Object, std::shared_ptr<Object>>(mod, "Object", doc::Object)
      .def_property_readonly(
          "id", [](Object* self) -> ObjectIDWrapper { return self->id(); },
          doc::Object_id)
      .def_property_readonly(
          "signature",
          [](Object* self) -> Signature { return self->meta().GetSignature(); },
          doc::Object_signature)
      .def_property_readonly("meta", &Object::meta,
                             py::return_value_policy::reference_internal,
                             doc::Object_meta)
      .def_property_readonly("nbytes", &Object::nbytes, doc::Object_nbytes)
      .def_property_readonly(
          "typename",
          [](const Object* object) -> const std::string {
            return object->meta().GetTypeName();
          },
          doc::Object_typename)
      .def(
          "member",
          [](Object const* self, std::string const& name) {
            return self->meta().GetMember(name);
          },
          doc::Object_member)
      .def(
          "__getitem__",
          [](Object const* self, std::string const& name) {
            return self->meta().GetMember(name);
          },
          doc::ObjectMeta__getitem__)
      .def_property_readonly("islocal", &Object::IsLocal, doc::Object_islocal)
      .def_property_readonly("ispersist", &Object::IsPersist,
                             doc::Object_ispersist)
      .def_property_readonly("isglobal", &Object::IsGlobal,
                             doc::Object_isglobal)
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
      .def(
          "seal",
          [](ObjectBuilder* self, py::object client) {
            std::shared_ptr<Object> object;
            Client* ipc_client = py::cast<Client*>(client.attr("ipc_client"));
            throw_on_error(self->Seal(*ipc_client, object));
            return object;
          },
          "client"_a)
      .def_property_readonly("issealed", &ObjectBuilder::sealed);
}

void bind_blobs(py::module& mod) {
  // Blob
  py::class_<Blob, std::shared_ptr<Blob>, Object>(
      mod, "Blob", py::buffer_protocol(), doc::Blob)
      .def_property_readonly("size", &Blob::size, doc::Blob_size)
      .def_property_readonly(
          "is_empty", [](Blob* self) { return self->allocated_size() == 0; },
          doc::Blob_is_empty)
      .def_static("empty", &Blob::MakeEmpty, doc::Blob_empty)
      .def("__len__", &Blob::size, doc::Blob__len__)
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
          [](Blob* self) { return reinterpret_cast<uintptr_t>(self->data()); },
          doc::Blob_address)
      .def_buffer([](Blob& blob) -> py::buffer_info {
        return py::buffer_info(const_cast<char*>(blob.data()), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, true);
      });

  // BlobBuilder
  py::class_<BlobWriter, std::shared_ptr<BlobWriter>, ObjectBuilder>(
      mod, "BlobBuilder", py::buffer_protocol(), doc::BlobBuilder)
      .def_property_readonly(
          "id", [](BlobWriter* self) -> ObjectIDWrapper { return self->id(); },
          doc::BlobBuilder_id)
      .def_property_readonly("size", &BlobWriter::size, doc::BlobBuilder__len__)
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
          "client"_a, doc::BlobBuilder_abort)
      .def(
          "shrink",
          [](BlobWriter* self, Client& client, const size_t size) {
            throw_on_error(self->Shrink(client, size));
          },
          "client"_a, "size"_a, doc::BlobBuilder_shrink)
      .def(
          "copy",
          [](BlobWriter* self, size_t const offset, uintptr_t ptr,
             size_t const size,
             size_t const concurrency = memory::default_memcpy_concurrency) {
            if (size == 0) {
              return;
            }
            memory::concurrent_memcpy(self->data() + offset,
                                      reinterpret_cast<void*>(ptr), size,
                                      concurrency);
          },
          "offset"_a, "address"_a, "size"_a,
          py::arg("concurrency") = memory::default_memcpy_concurrency,
          doc::BlobBuilder_copy)
      .def(
          "copy",
          [](BlobWriter* self, size_t offset, py::buffer const& buffer,
             size_t const concurrency = memory::default_memcpy_concurrency) {
            if (self->size() == 0) {
              return;
            }
            throw_on_error(copy_memoryview(self->data(), self->size(),
                                           buffer.ptr(), offset, concurrency));
          },
          "offset"_a, "buffer"_a,
          py::arg("concurrency") = memory::default_memcpy_concurrency,
          doc::BlobBuilder_copy)
      .def(
          "copy",
          [](BlobWriter* self, size_t offset, py::bytes const& bs,
             size_t const concurrency = memory::default_memcpy_concurrency) {
            char* buffer = nullptr;
            ssize_t size = 0;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bs.ptr(), &buffer, &size)) {
              py::pybind11_fail("Unable to extract bytes contents!");
            }
            if (size == 0) {
              return;
            }
            if (offset + size > self->size()) {
              throw_on_error(Status::AssertionFailed(
                  "Expect a source buffer with size at most '" +
                  std::to_string(self->size() - offset) +
                  "', but the buffer size is '" + std::to_string(size) + "'"));
            }
            memory::concurrent_memcpy(self->data() + offset, buffer, size,
                                      concurrency);
          },
          "offset"_a, "bytes"_a,
          py::arg("concurrency") = memory::default_memcpy_concurrency,
          doc::BlobBuilder_copy)
      .def_property_readonly(
          "address",
          [](BlobWriter* self) {
            return reinterpret_cast<uintptr_t>(self->data());
          },
          doc::BlobBuilder_address)
      .def_buffer([](BlobWriter& blob) -> py::buffer_info {
        return py::buffer_info(blob.data(), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, false);
      });

  // RemoteBlob
  py::class_<RemoteBlob, std::shared_ptr<RemoteBlob>, Object>(
      mod, "RemoteBlob", py::buffer_protocol(), doc::RemoteBlob)
      .def_property_readonly(
          "id", [](RemoteBlob* self) -> ObjectIDWrapper { return self->id(); },
          doc::RemoteBlob_id)
      .def_property_readonly(
          "instance_id",
          [](RemoteBlob* self) -> InstanceID { return self->instance_id(); },
          doc::RemoteBlob_instance_id)
      .def_property_readonly("size", &RemoteBlob::size, doc::RemoteBlob__len__)
      .def_property_readonly("allocated_size", &RemoteBlob::allocated_size)
      .def_property_readonly(
          "is_empty",
          [](RemoteBlob* self) { return self->allocated_size() == 0; },
          doc::RemoteBlob_is_empty)
      .def("__len__", &RemoteBlob::allocated_size, doc::RemoteBlob__len__)
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
      .def_property_readonly(
          "address",
          [](RemoteBlob* self) {
            return reinterpret_cast<uintptr_t>(self->data());
          },
          doc::RemoteBlob_address)
      .def_buffer([](RemoteBlob& blob) -> py::buffer_info {
        return py::buffer_info(const_cast<char*>(blob.data()), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, true);
      });

  // RemoteBlobBuilder
  py::class_<RemoteBlobWriter, std::shared_ptr<RemoteBlobWriter>>(
      mod, "RemoteBlobBuilder", py::buffer_protocol(), doc::RemoteBlobBuilder)
      .def(py::init<>(
               [](const size_t size) -> std::shared_ptr<RemoteBlobWriter> {
                 return std::make_shared<RemoteBlobWriter>(size);
               }),
           py::arg("size"))
      .def_static(
          "make",
          [](const size_t size) -> std::shared_ptr<RemoteBlobWriter> {
            return RemoteBlobWriter::Make(size);
          },
          "size"_a)
      .def_static(
          "wrap",
          [](uintptr_t const data,
             const size_t size) -> std::shared_ptr<RemoteBlobWriter> {
            return RemoteBlobWriter::Wrap(
                reinterpret_cast<const uint8_t*>(data), size);
          },
          "data"_a, "size"_a)
      .def_static(
          "wrap",
          [](py::buffer const& buffer,
             const size_t nbytes) -> std::shared_ptr<RemoteBlobWriter> {
            return RemoteBlobWriter::Wrap(
                reinterpret_cast<const uint8_t*>(buffer.ptr()), nbytes);
          },
          "data"_a, "size"_a)
      .def_static(
          "wrap",
          [](py::bytes const& bs) -> std::shared_ptr<RemoteBlobWriter> {
            char* buffer = nullptr;
            ssize_t size = 0;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bs.ptr(), &buffer, &size)) {
              py::pybind11_fail("Unable to extract bytes contents!");
            }
            return RemoteBlobWriter::Wrap(
                reinterpret_cast<const uint8_t*>(buffer), size);
          },
          "data"_a)
      .def_property_readonly("size", &RemoteBlobWriter::size,
                             doc::RemoteBlobBuilder_size)
      .def("__len__", &RemoteBlobWriter::size, doc::RemoteBlobBuilder_size)
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
          "client"_a, doc::RemoteBlobBuilder_abort)
      .def(
          "copy",
          [](RemoteBlobWriter* self, size_t const offset, uintptr_t ptr,
             size_t const size,
             size_t const concurrency = memory::default_memcpy_concurrency) {
            if (size == 0) {
              return;
            }
            memory::concurrent_memcpy(self->data() + offset,
                                      reinterpret_cast<void*>(ptr), size,
                                      concurrency);
          },
          "offset"_a, "address"_a, "size"_a,
          py::arg("concurrency") = memory::default_memcpy_concurrency,
          doc::RemoteBlobBuilder_copy)
      .def(
          "copy",
          [](RemoteBlobWriter* self, size_t offset, py::buffer const& buffer,
             size_t const concurrency = memory::default_memcpy_concurrency) {
            if (self->size() == 0) {
              return;
            }
            throw_on_error(copy_memoryview(self->data(), self->size(),
                                           buffer.ptr(), offset, concurrency));
          },
          "offset"_a, "buffer"_a,
          py::arg("concurrency") = memory::default_memcpy_concurrency,
          doc::RemoteBlobBuilder_copy)
      .def(
          "copy",
          [](RemoteBlobWriter* self, size_t offset, py::bytes const& bs,
             size_t const concurrency = memory::default_memcpy_concurrency) {
            char* buffer = nullptr;
            ssize_t size = 0;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bs.ptr(), &buffer, &size)) {
              py::pybind11_fail("Unable to extract bytes contents!");
            }
            if (size == 0) {
              return;
            }
            if (offset + size > self->size()) {
              throw_on_error(Status::AssertionFailed(
                  "Expect a source buffer with size at most '" +
                  std::to_string(self->size() - offset) +
                  "', but the buffer size is '" + std::to_string(size) + "'"));
            }
            throw_on_error(copy_memoryview(self->data(), self->size(), buffer,
                                           size, offset, concurrency));
          },
          "offset"_a, "bytes"_a,
          py::arg("concurrency") = memory::default_memcpy_concurrency,
          doc::RemoteBlobBuilder_copy)
      .def_property_readonly(
          "address",
          [](RemoteBlobWriter* self) {
            return reinterpret_cast<uintptr_t>(self->data());
          },
          doc::RemoteBlobBuilder_address)
      .def_buffer([](RemoteBlobWriter& blob) -> py::buffer_info {
        return py::buffer_info(blob.data(), sizeof(int8_t),
                               py::format_descriptor<int8_t>::format(), 1,
                               {blob.size()}, {sizeof(int8_t)}, false);
      });
}

}  // namespace vineyard
