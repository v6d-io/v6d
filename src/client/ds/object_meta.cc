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
  meta_.put("id", VYObjectIDToString(id));
}

const ObjectID ObjectMeta::GetId() const {
  return VYObjectIDFromString(meta_.get<std::string>("id"));
}

void ObjectMeta::SetTypeName(const std::string& type_name) {
  meta_.put("typename", type_name);
}

std::string const ObjectMeta::GetTypeName() const {
  return meta_.get<std::string>("typename");
}

void ObjectMeta::SetNBytes(const size_t nbytes) { meta_.put("nbytes", nbytes); }

size_t const ObjectMeta::GetNBytes() const {
  auto nbytes = meta_.get_optional<size_t>("nbytes");
  if (nbytes) {
    return nbytes.get();
  } else {
    // return -1 to indicate such objects has no nbytes, e.g., global objects
    return -1;
  }
}

InstanceID const ObjectMeta::GetInstanceId() const {
  return meta_.get<InstanceID>("instance_id");
}

bool const ObjectMeta::IsLocal() const {
  if (client_ && meta_.find("instance_id") != meta_.not_found()) {
    return client_->instance_id() == GetInstanceId();
  } else {
    // it is a newly created metadata
    return true;
  }
}

bool const ObjectMeta::Haskey(std::string const& key) const {
  return meta_.find(key) != meta_.not_found();
}

void ObjectMeta::AddKeyValue(const std::string& key, const std::string& value) {
  meta_.put(key, value);
}

void ObjectMeta::AddKeyValue(const std::string& key, const ptree& value) {
  std::stringstream ss;
  boost::property_tree::json_parser::write_json(ss, value, false);
  meta_.put(key, ss.str());
}

void ObjectMeta::GetKeyValue(const std::string& key, ptree& value) const {
  std::istringstream is(meta_.get<std::string>(key));
  try {
    boost::property_tree::read_json(is, value);
  } catch (...) {}  // just ignore the boost ptree exception to prevent crash.
}

void ObjectMeta::AddMember(const std::string& name, const ObjectMeta& member) {
  VINEYARD_ASSERT(meta_.find(name) == meta_.not_found());
  meta_.add_child(name, member.meta_);
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
  VINEYARD_ASSERT(meta_.find(name) == meta_.not_found());
  ptree member_node;
  member_node.add("id", VYObjectIDToString(member_id));
  meta_.add_child(name, member_node);
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
  auto const& child_meta = meta_.get_child_optional(name);
  VINEYARD_ASSERT(child_meta);
  ret.SetClient(client_);

#if !defined(NDEBUG)  // slow path, but accurate
  ret.SetMetaData(this->client_, child_meta.get());
  auto const& all_blobs = blob_set_->AllBlobs();
  for (auto const& id : ret.blob_set_->AllBlobIds()) {
    auto iter = all_blobs.find(id);
    // for remote object, the blob may not present here
    if (iter != all_blobs.end()) {
      ret.SetBlob(id, iter->second.BufferUnsafe());
    }
  }
#else  // fast path
  ret.meta_ = child_meta.get();
  ret.blob_set_ = this->blob_set_;
#endif
  return ret;
}

void ObjectMeta::PrintMeta() const {
  thread_local std::stringstream ss;
  ss.str("");
  ss.clear();
  bpt::write_json(ss, meta_, true);
  LOG(INFO) << ss.str();
}

const bool ObjectMeta::incomplete() const { return incomplete_; }

const ptree& ObjectMeta::MetaData() const { return meta_; }

ptree& ObjectMeta::MutMetaData() { return meta_; }

void ObjectMeta::SetMetaData(ClientBase* client, const ptree& meta) {
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

void ObjectMeta::findAllBlobs(const ptree& tree, InstanceID const instance_id) {
  if (tree.empty()) {
    return;
  }
  ObjectID member_id = VYObjectIDFromString(tree.get<std::string>("id"));
  if (IsBlob(member_id)) {
    blob_set_->EmplaceId(member_id, tree.get<size_t>("length"),
                         tree.get<InstanceID>("instance_id") == instance_id);
  } else {
    for (ptree::const_iterator it = tree.begin(); it != tree.end(); ++it) {
      if (!it->second.empty()) {
        this->findAllBlobs(it->second, instance_id);
      }
    }
  }
}

void ObjectMeta::SetInstanceId(const InstanceID instance_id) {
  meta_.put("instance_id", instance_id);
}

}  // namespace vineyard
