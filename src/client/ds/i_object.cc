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

#include "client/ds/i_object.h"

#include "client/client.h"
#include "client/client_base.h"

namespace vineyard {

Object::~Object() {}

const ObjectID Object::id() const { return id_; }

const ObjectMeta& Object::meta() const { return meta_; }

size_t const Object::nbytes() const { return meta_.GetNBytes(); }

std::shared_ptr<Object> Object::_Seal(Client& client) {
  return shared_from_this();
}

void Object::Construct(const ObjectMeta& meta) {
  this->meta_ = meta;
  this->id_ = meta.GetId();
}

Status Object::Persist(ClientBase& client) const {
  return client.Persist(this->id_);
}

bool const Object::IsLocal() const { return meta_.IsLocal(); }

bool const Object::IsPersist() const {
  bool persist = !this->meta_.GetKeyValue<bool>("transient");
  if (!persist) {
    VINEYARD_CHECK_OK(this->meta_.GetClient()->IfPersist(this->id_, persist));
    // as an optimization: cache the query result
    if (persist) {
      this->meta_.AddKeyValue("transient", false);
    }
  }
  return persist;
}

bool const Object::IsGlobal() const { return meta_.IsGlobal(); }

std::shared_ptr<Object> ObjectBuilder::Seal(Client& client) {
  auto object = this->_Seal(client);
  VINEYARD_CHECK_OK(client.PostSeal(object->meta()));
  return object;
}

}  // namespace vineyard
