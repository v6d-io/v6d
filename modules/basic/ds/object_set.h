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

#ifndef MODULES_BASIC_DS_OBJECT_SET_H_
#define MODULES_BASIC_DS_OBJECT_SET_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic/ds/object_set.h"
#include "client/client.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

class ObjectSetBuilder;

class ObjectSet : public Registered<ObjectSet> {
 public:
  static std::shared_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(std::make_shared<ObjectSet>());
  }

  const size_t InstanceNum() const { return this->num_of_instances_; }

  const size_t ObjectNum() const { return this->num_of_objects_; }

  const std::shared_ptr<Object>& ObjectAt(InstanceID instance_id) const {
    return *objects_.at(instance_id).begin();
  }

  const std::vector<std::shared_ptr<Object>>& ObjectsAt(
      InstanceID instance_id) const {
    return objects_.at(instance_id);
  }

  void Construct(const ObjectMeta& meta) override {
    std::string __type_name = type_name<ObjectSet>();
    CHECK(meta.GetTypeName() == __type_name);
    this->meta_ = meta;
    this->id_ = meta.GetId();

    meta.GetKeyValue("num_of_instances", num_of_instances_);
    meta.GetKeyValue("num_of_objects", num_of_objects_);

    for (size_t idx = 0; idx < num_of_objects_; ++idx) {
      auto object = meta.GetMember("object_" + std::to_string(idx));
      InstanceID instance_id = object->meta().GetInstanceId();
      objects_[instance_id].emplace_back(object);
    }
  }

 private:
  size_t num_of_instances_;
  size_t num_of_objects_;
  std::unordered_map<InstanceID, std::vector<std::shared_ptr<Object>>> objects_;

  friend class ObjectSetBuilder;
};

class ObjectSetBuilder : public ObjectBuilder {
 public:
  explicit ObjectSetBuilder(Client& client) {}

  const size_t ObjectNum() const { return this->num_of_objects_; }

  const size_t InstanceNum() const { return this->instance_ids_.size(); }

  void AddObject(InstanceID instance_id, ObjectID object_id) {
    this->object_ids_.emplace_back(object_id);
    this->instance_ids_.emplace(instance_id);
    this->num_of_objects_++;
  }

  void AddObjects(InstanceID instance_id,
                  std::vector<ObjectID> const& object_ids) {
    this->object_ids_.insert(object_ids_.end(), object_ids.begin(),
                             object_ids.end());
    this->instance_ids_.emplace(instance_id);
    this->num_of_objects_ += object_ids.size();
  }

  Status Build(Client& client) override { return Status::OK(); }

 protected:
  std::shared_ptr<Object> _Seal(Client& client) override {
    auto object_set = std::make_shared<ObjectSet>();

    // Use instances that has this object, rather than total instances.
    size_t num_of_instances = instance_ids_.size();

    object_set->num_of_instances_ = num_of_instances;
    object_set->num_of_objects_ = num_of_objects_;

    object_set->meta_.SetTypeName(type_name<ObjectSet>());
    object_set->meta_.SetGlobal(true);
    object_set->meta_.SetNBytes(-1 /* FIXME */);

    object_set->meta_.AddKeyValue("num_of_instances", num_of_instances);
    object_set->meta_.AddKeyValue("num_of_objects", num_of_objects_);

    // NB: the instance_id may be not consecutive.

    size_t idx = 0;
    for (auto const& object_id : object_ids_) {
      object_set->meta_.AddMember("object_" + std::to_string(idx), object_id);
      idx += 1;
    }

    VINEYARD_CHECK_OK(
        client.CreateMetaData(object_set->meta_, object_set->id_));
    return std::static_pointer_cast<Object>(object_set);
  }

  size_t num_of_objects_ = 0;
  std::set<InstanceID> instance_ids_;
  std::vector<ObjectID> object_ids_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_OBJECT_SET_H_
