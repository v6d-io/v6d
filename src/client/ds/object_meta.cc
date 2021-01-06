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

#include "client/ds/object_meta.h"

#include "client/client.h"
#include "client/ds/blob.h"

namespace vineyard {

ObjectMeta::ObjectMeta() : blob_set_(std::make_shared<BlobSet>()) {}

void ObjectMeta::SetClient(ClientBase* client) { this->client_ = client; }

ClientBase* ObjectMeta::GetClient() const { return client_; }

void ObjectMeta::SetId(const ObjectID& id) {
  meta_["id"] = VYObjectIDToString(id);
}

const ObjectID ObjectMeta::GetId() const {
  return VYObjectIDFromString(meta_["id"].get_ref<std::string const&>());
}

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
  auto instance_id = meta_["instance_id"];
  if (instance_id.is_null()) {
    // it is a newly created metadata
    return true;
  } else {
    return client_->instance_id() == instance_id.get<InstanceID>();
  }
}

bool const ObjectMeta::Haskey(std::string const& key) const {
  return meta_.contains(key);
}

void ObjectMeta::AddKeyValue(const std::string& key, const std::string& value) {
  meta_[key] = value;
}

void ObjectMeta::AddKeyValue(const std::string& key, const json& value) {
  meta_[key] = json_to_string(value);
}

void ObjectMeta::GetKeyValue(const std::string& key, json& value) const {
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
  this->blob_set_->Extend(member.blob_set_);
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
  member_node["id"] = VYObjectIDToString(member_id);
  meta_[name] = member_node;
  // mark the meta_ as incomplete
  incomplete_ = true;
}

std::shared_ptr<Object> ObjectMeta::GetMember(const std::string& name) const {
  ObjectMeta meta = this->GetMemberMeta(name);
  auto object = ObjectFactory::Create(meta.GetTypeName());
  if (object == nullptr) {
    object = std::shared_ptr<Object>(new Object());
  }
  object->Construct(meta);
  return object;
}

ObjectMeta ObjectMeta::GetMemberMeta(const std::string& name) const {
  ObjectMeta ret;
  auto const& child_meta = meta_[name];
  VINEYARD_ASSERT(!child_meta.is_null(), "Failed to get member " + name);
  ret.SetClient(client_);

#if !defined(NDEBUG)  // slow path, but accurate
  ret.SetMetaData(this->client_, child_meta);
  auto const& all_blobs = blob_set_->AllBlobs();
  for (auto const& id : ret.blob_set_->AllBlobIds()) {
    auto iter = all_blobs.find(id);
    // for remote object, the blob may not present here
    if (iter != all_blobs.end()) {
      ret.SetBlob(id, iter->second.BufferUnsafe());
    }
  }
#else  // fast path
  ret.meta_ = child_meta;
  ret.blob_set_ = this->blob_set_;
#endif
  return ret;
}

void ObjectMeta::PrintMeta() const { LOG(INFO) << meta_.dump(4); }

const bool ObjectMeta::incomplete() const { return incomplete_; }

const json& ObjectMeta::MetaData() const { return meta_; }

json& ObjectMeta::MutMetaData() { return meta_; }

void ObjectMeta::SetMetaData(ClientBase* client, const json& meta) {
  this->client_ = client;
  this->meta_ = meta;
  findAllBlobs(meta_, this->client_->instance_id());
}

const std::shared_ptr<BlobSet>& ObjectMeta::GetBlobSet() const {
  return blob_set_;
}

void ObjectMeta::SetBlob(const ObjectID& id,
                         const std::shared_ptr<arrow::Buffer>& buffer) {
  // After `findAllBlobs` we know the blob set of this object. If the given id
  // is not present in the blob set, it should be an error.
  VINEYARD_ASSERT(blob_set_->Contains(id));
  blob_set_->EmplaceBlob(id, buffer);
}

void ObjectMeta::findAllBlobs(const json& tree, InstanceID const instance_id) {
  if (tree.empty()) {
    return;
  }
  ObjectID member_id =
      VYObjectIDFromString(tree["id"].get_ref<std::string const&>());
  if (IsBlob(member_id)) {
    blob_set_->EmplaceId(member_id, tree["length"].get<size_t>(),
                         tree["instance_id"].get<InstanceID>() == instance_id);
  } else {
    for (auto& item : tree) {
      if (item.is_object()) {
        this->findAllBlobs(item, instance_id);
      }
    }
  }
}

void ObjectMeta::SetInstanceId(const InstanceID instance_id) {
  meta_["instance_id"] = instance_id;
}

template <>
const std::string ObjectMeta::GetKeyValue<std::string>(
    const std::string& key) const {
  return meta_[key].get_ref<const std::string&>();
}

template <>
const json ObjectMeta::GetKeyValue<json>(const std::string& key) const {
  try {
    return json::parse(meta_[key].get_ref<const std::string&>());
  } catch (nlohmann::json::parse_error const&) {
    throw std::out_of_range("Invalid json value at key '" + key +
                            "': " + meta_[key].get_ref<const std::string&>());
  }
}

}  // namespace vineyard
