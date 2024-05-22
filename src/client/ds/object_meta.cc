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

#include "client/ds/object_meta.h"

#include <iostream>

#include "client/client.h"
#include "client/client_base.h"
#include "client/ds/blob.h"
#include "common/util/env.h"

namespace vineyard {

ObjectMeta::ObjectMeta() : buffer_set_(std::make_shared<BufferSet>()) {}

ObjectMeta::~ObjectMeta() {}

ObjectMeta::ObjectMeta(const ObjectMeta& other) {
  this->client_ = other.client_;
  this->meta_ = other.meta_;
  this->buffer_set_ = other.buffer_set_;
  this->incomplete_ = other.incomplete_;
  this->force_local_ = other.force_local_;
}

ObjectMeta& ObjectMeta::operator=(ObjectMeta const& other) {
  this->client_ = other.client_;
  this->meta_ = other.meta_;
  this->buffer_set_ = other.buffer_set_;
  this->incomplete_ = other.incomplete_;
  this->force_local_ = other.force_local_;
  return *this;
}

void ObjectMeta::SetClient(ClientBase* client) { this->client_ = client; }

ClientBase* ObjectMeta::GetClient() const { return client_; }

void ObjectMeta::SetId(const ObjectID& id) {
  meta_["id"] = ObjectIDToString(id);
}

const ObjectID ObjectMeta::GetId() const {
  return ObjectIDFromString(meta_["id"].get_ref<std::string const&>());
}

const Signature ObjectMeta::GetSignature() const {
  return meta_["signature"].get<Signature>();
}

void ObjectMeta::ResetSignature() { this->ResetKey("signature"); }

void ObjectMeta::SetGlobal(bool global) { meta_["global"] = global; }

const bool ObjectMeta::IsGlobal() const { return meta_.value("global", false); }

void ObjectMeta::SetTypeName(const std::string& type_name) {
  meta_["typename"] = type_name;
}

std::string const& ObjectMeta::GetTypeName() const {
  return meta_["typename"].get_ref<std::string const&>();
}

void ObjectMeta::SetNBytes(const size_t nbytes) { meta_["nbytes"] = nbytes; }

size_t const ObjectMeta::GetNBytes() const {
  auto nbytes = meta_["nbytes"];
  if (nbytes.is_null()) {
    // return 0 to indicate such objects has no nbytes, e.g., global objects
    return 0;
  } else {
    return nbytes.get<size_t>();
  }
}

InstanceID const ObjectMeta::GetInstanceId() const {
  return meta_["instance_id"].get<InstanceID>();
}

bool const ObjectMeta::IsLocal() const {
  if (this->force_local_) {
    return true;
  }
  auto instance_id = meta_["instance_id"];
  if (instance_id.is_null()) {
    // it is a newly created metadata
    return true;
  } else {
    if (client_) {
      return client_->instance_id() == instance_id.get<InstanceID>();
    } else {
      return false;
    }
  }
}

void ObjectMeta::ForceLocal() const { this->force_local_ = true; }

bool const ObjectMeta::Haskey(std::string const& key) const {
  return HasKey(key);
}

bool const ObjectMeta::HasKey(std::string const& key) const {
  return meta_.contains(key);
}

void ObjectMeta::ResetKey(std::string const& key) {
  if (meta_.contains(key)) {
    meta_.erase(key);
  }
}

void ObjectMeta::AddKeyValue(const std::string& key, const std::string& value) {
  meta_[key] = value;
}

void ObjectMeta::AddKeyValue(const std::string& key, const json& value) {
  meta_[key] = json_to_string(value);
}

template <>
const json ObjectMeta::GetKeyValue<json>(const std::string& key) const {
  json value;
  GetKeyValue(key, value);
  return value;
}

void ObjectMeta::GetKeyValue(const std::string& key, json& value) const {
  if (!HasKey(key)) {
    value = json::object();
    return;
  }
  try {
    value = json::parse(meta_[key].get_ref<const std::string&>());
  } catch (nlohmann::json::parse_error const&) {
    throw std::out_of_range("Invalid json value at key '" + key +
                            "': " + meta_[key].get_ref<const std::string&>());
  }
}

void ObjectMeta::AddMember(const std::string& name, const ObjectMeta& member) {
  VINEYARD_ASSERT(!meta_.contains(name));
  meta_[name] = member.meta_;
  this->buffer_set_->Extend(member.buffer_set_);
}

void ObjectMeta::AddMember(const std::string& name, const Object& member) {
  this->AddMember(name, member.meta());
}

void ObjectMeta::AddMember(const std::string& name, const Object* member) {
  this->AddMember(name, member->meta());
}

void ObjectMeta::AddMember(const std::string& name,
                           const std::shared_ptr<Object>& member) {
  this->AddMember(name, member->meta());
}

void ObjectMeta::AddMember(const std::string& name, const ObjectID member_id) {
  VINEYARD_ASSERT(!meta_.contains(name));
  json member_node;
  member_node["id"] = ObjectIDToString(member_id);
  meta_[name] = member_node;
  // mark the meta_ as incomplete
  incomplete_ = true;
}

std::shared_ptr<Object> ObjectMeta::GetMember(const std::string& name) const {
  ObjectMeta meta = this->GetMemberMeta(name);
  auto object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::unique_ptr<Object>(new Object());
  }
  object->Construct(meta);
  return object;
}

Status ObjectMeta::GetMember(const std::string& name,
                             std::shared_ptr<Object>& object) const {
  ObjectMeta meta;
  RETURN_ON_ERROR(GetMemberMeta(name, meta));
  RETURN_ON_ASSERT(!meta.MetaData().empty(), "metadata shouldn't be empty");
  object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::unique_ptr<Object>(new Object());
  }
  object->Construct(meta);
  return Status::OK();
}

ObjectMeta ObjectMeta::GetMemberMeta(const std::string& name) const {
  ObjectMeta meta;
  VINEYARD_CHECK_OK(GetMemberMeta(name, meta));
  return meta;
}

Status ObjectMeta::GetMemberMeta(const std::string& name,
                                 ObjectMeta& meta) const {
  auto const& child_meta = meta_[name];
  RETURN_ON_ASSERT(!child_meta.is_null(),
                   "Failed to get member '" + name + "'");

  meta.Reset();
  meta.SetMetaData(this->client_, child_meta);
  auto const& all_blobs = buffer_set_->AllBuffers();
  for (auto const& blob : meta.buffer_set_->AllBuffers()) {
    auto iter = all_blobs.find(blob.first);
    // for remote object, the blob may not present here
    if (iter != all_blobs.end()) {
      meta.SetBuffer(blob.first, iter->second);
    }
  }
  if (this->force_local_) {
    meta.ForceLocal();
  }
  return Status::OK();
}

Status ObjectMeta::GetBuffer(const ObjectID blob_id,
                             std::shared_ptr<Buffer>& buffer) const {
  if (buffer_set_->Get(blob_id, buffer)) {
    return Status::OK();
  } else {
    return Status::ObjectNotExists(
        "The target blob " + ObjectIDToString(blob_id) + " doesn't exist");
  }
}

void ObjectMeta::SetBuffer(const ObjectID& id,
                           const std::shared_ptr<Buffer>& buffer) {
  // After `findAllBlobs` we know the buffer set of this object. If the given id
  // is not present in the buffer set, it should be an error.
  VINEYARD_ASSERT(buffer_set_->Contains(id));
  VINEYARD_CHECK_OK(buffer_set_->EmplaceBuffer(id, buffer));
}

void ObjectMeta::Reset() {
  client_ = nullptr;
  meta_ = json::object();
  buffer_set_.reset(new BufferSet());
  incomplete_ = false;
}

size_t ObjectMeta::MemoryUsage() const {
  size_t total_size = 0;
  for (auto const& item : buffer_set_->AllBuffers()) {
    if (item.second) {
      total_size += item.second->size();
    }
  }
  return total_size;
}

size_t ObjectMeta::MemoryUsage(json& usages, const bool pretty) const {
  std::function<size_t(const json&, json&)> traverse =
      [this, pretty, &traverse](const json& tree, json& usages) -> size_t {
    if (!tree.is_object() || tree.empty()) {
      return 0;
    }
    ObjectID member_id =
        ObjectIDFromString(tree["id"].get_ref<std::string const&>());
    if (IsBlob(member_id)) {
      std::shared_ptr<Buffer> buffer;
      if (buffer_set_->Get(member_id, buffer)) {
        if (pretty) {
          usages = json::string_t(prettyprint_memory_size(buffer->size()));
          return buffer->size();
        } else {
          usages = json::number_integer_t(buffer->size());
          return buffer->size();
        }
      } else {
        return 0;
      }
    } else {
      size_t total = 0;
      for (auto& item : tree.items()) {
        if (item.value().is_object()) {
          json usage;
          total += traverse(item.value(), usage);
          if (!usage.is_null()) {
            usages[item.key()] = usage;
          }
        }
      }
      if (pretty) {
        usages["summary"] = json::string_t(prettyprint_memory_size(total));
      } else {
        usages["summary"] = json::number_integer_t(total);
      }
      return total;
    }
  };
  return traverse(this->meta_, usages);
}

uint64_t ObjectMeta::Timestamp() const {
  return meta_.value("__timestamp", static_cast<uint64_t>(0));
}

json ObjectMeta::Labels() const {
  const std::string label_string = meta_.value("__labels", "{}");
  Status s;
  json labels;
  CATCH_JSON_ERROR(labels, s, json::parse(label_string));
  return labels;
}

const std::string ObjectMeta::Label(const std::string& key) const {
  auto labels = Labels();
  return labels.value(key, "");
}

std::string ObjectMeta::ToString() const { return meta_.dump(4); }

void ObjectMeta::PrintMeta() const { std::clog << meta_.dump(4) << std::endl; }

const bool ObjectMeta::incomplete() const { return incomplete_; }

const json& ObjectMeta::MetaData() const { return meta_; }

json& ObjectMeta::MutMetaData() { return meta_; }

void ObjectMeta::SetMetaData(ClientBase* client, const json& meta) {
  this->client_ = client;
  this->meta_ = meta;

  std::function<void(const json&)> traverse = [this,
                                               &traverse](const json& tree) {
    if (!tree.is_object() || tree.empty()) {
      return;
    }
    ObjectID member_id =
        ObjectIDFromString(tree["id"].get_ref<std::string const&>());
    if (IsBlob(member_id)) {
      if (client_ == nullptr) {
        VINEYARD_CHECK_OK(buffer_set_->EmplaceBuffer(member_id));
      } else {
        InstanceID instance_id = tree["instance_id"].get<InstanceID>();
        if ((client_->IsIPC() && instance_id == client_->instance_id()) ||
            (client_->IsRPC() &&
             instance_id == client_->remote_instance_id())) {
          VINEYARD_CHECK_OK(buffer_set_->EmplaceBuffer(member_id));
        }
      }
    } else {
      for (auto& item : tree) {
        if (item.is_object()) {
          traverse(item);
        }
      }
    }
  };
  traverse(this->meta_);
}

std::unique_ptr<ObjectMeta> ObjectMeta::Unsafe(std::string meta,
                                               size_t nobjects,
                                               ObjectID* objects,
                                               uintptr_t* pointers,
                                               size_t* sizes) {
  try {
    return ObjectMeta::Unsafe(json::parse(meta), nobjects, objects, pointers,
                              sizes);
  } catch (std::exception const&) { return nullptr; }
}

std::unique_ptr<ObjectMeta> ObjectMeta::Unsafe(json meta, size_t nobjects,
                                               ObjectID* objects,
                                               uintptr_t* pointers,
                                               size_t* sizes) {
  std::unique_ptr<ObjectMeta> metadata(new ObjectMeta());
  metadata->SetMetaData(nullptr, meta);
  for (size_t index = 0; index < nobjects; ++index) {
    auto buffer = std::make_shared<Buffer>(
        reinterpret_cast<const uint8_t*>(pointers[index]), sizes[index]);
    metadata->SetBuffer(objects[index], buffer);
  }
  return metadata;
}

const std::shared_ptr<BufferSet>& ObjectMeta::GetBufferSet() const {
  return buffer_set_;
}

void ObjectMeta::SetInstanceId(const InstanceID instance_id) {
  meta_["instance_id"] = instance_id;
}

void ObjectMeta::SetSignature(const Signature signature) {
  meta_["signature"] = signature;
}

}  // namespace vineyard
