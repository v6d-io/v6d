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

#ifndef SRC_CLIENT_DS_COLLECTION_H_
#define SRC_CLIENT_DS_COLLECTION_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

template <typename T>
class Collection;

template <typename T>
struct collection_type {
  using type = Collection<T>;
};

template <typename T>
using collection_type_t = typename collection_type<T>::type;

namespace detail {
inline std::string index_to_key(const size_t index) {
  return "partitions_-" + std::to_string(index);
}

inline int64_t index_from_key(const std::string& key) {
  if (key.substr(0, 11) != "partitions_-") {
    return -1;
  }
  size_t consumed = 0;
  int64_t index = std::stol(key.substr(11), &consumed);
  if (consumed != key.size() - 11) {
    return -1;
  }
  return index;
}
}  // namespace detail

/**
 * Collection has been class `GlobalObject`, but it not necessary a "global"
 * object.
 */
template <typename T>
class Collection : public Registered<Collection<T>>, public GlobalObject {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<Collection<T>>{new Collection<T>()});
  }

  struct iterator {
    iterator() {}

    iterator(const Collection<T>& collection, const size_t index)
        : collection_(collection), index_(index) {}

    iterator& operator++() {
      if (index_ >= collection_.size()) {
        throw std::out_of_range("index out of range");
      }
      do {
        ++index_;
      } while (!Exists() && index_ < collection_.size());
      return *this;
    }

    iterator operator++(int) {
      iterator copy(*this);
      operator++();
      return copy;
    }

    bool Exists() const {
      return this->collection_.meta_.HasKey(detail::index_to_key(index_));
    }

    bool IsLocal() const {
      if (index_ >= collection_.size()) {
        return false;
      }
      ObjectMeta meta;
      auto s = this->collection_.meta_.GetMemberMeta(
          detail::index_to_key(index_), meta);
      if (s.ok()) {
        return meta.IsLocal();
      } else {
        return false;
      }
    }

    iterator& Next() { return operator++(); }

    iterator& NextLocal() {
      do {
        operator++();
      } while (!IsLocal() && index_ < collection_.size());
      return *this;
    }

    bool HasNext() const { return index_ + 1 < collection_.size(); }

    bool HasNextLocal() const {
      size_t next = index_ + 1;
      while (next < collection_->size()) {
        ObjectMeta meta;
        auto s = this->collection_.meta_.GetMemberMeta(
            detail::index_to_key(next), meta);
        if (s.ok()) {
          if (meta.IsLocal()) {
            return true;
          }
        }
        ++next;
      }
      return false;
    }

    const std::shared_ptr<T> operator*() const {
      if (index_ >= collection_.size()) {
        throw std::out_of_range("index out of range");
      }
      std::shared_ptr<T> result = nullptr;
      auto s = this->collection_.meta_.template GetMember<T>(
          detail::index_to_key(index_), result);
      if (s.ok()) {
        return result;
      } else {
        return nullptr;
      }
    }

    std::shared_ptr<T> operator->() const { return operator*(); }

    bool operator==(const iterator& other) const {
      return collection_.id() == other.collection_.id() &&
             index_ == other.index_;
    }

    bool operator!=(const iterator& other) const { return !(*this == other); }

   private:
    const Collection<T>& collection_;
    size_t index_;
    bool filter_local_ = false;
  };

  void Construct(const ObjectMeta& meta) override {
    std::string __type_name = type_name<collection_type_t<T>>();
    VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                    "Expect typename '" + __type_name + "', but got '" +
                        meta.GetTypeName() + "'");
    Object::Construct(meta);
    this->meta_.GetKeyValue("params_", this->params_);
    this->meta_.GetKeyValue("partitions_-size", this->chunk_size_);
  }

  size_t size() const { return this->chunk_size_; }

  std::map<std::string, std::string> const& GetParams() const {
    return this->params_;
  }

  template <typename Value>
  Status GetKeyValue(std::string const& key, T& value) const {
    return this->meta_.GetKeyValue(key, value);
  }

  Status GetMember(std::string const& key,
                   std::shared_ptr<Object>& object) const {
    return this->meta_.GetMember(key, object);
  }

  template <typename O>
  Status GetMember(std::string const& key, std::shared_ptr<O>& object) const {
    return this->meta_.GetMember(key, object);
  }

  Status GetMember(const size_t index, std::shared_ptr<Object>& object) const {
    return this->meta_.GetMember(detail::index_to_key(index), object);
  }

  template <typename O>
  Status GetMember(const size_t index, std::shared_ptr<O>& object) const {
    return this->meta_.GetMember(detail::index_to_key(index), object);
  }

  const iterator Begin() const { return iterator(*this, 0); }

  const iterator End() const { return iterator(*this, this->size()); }

  /**
   * @brief Get the local partitions of the vineyard instance that is
   * connected from the client.
   *
   * @param client The client connected to a vineyard instance.
   * @return The vector of pointers to the local partitions.
   */
  const iterator LocalBegin() const {
    iterator iter = iterator(*this, 0);
    if (!iter.IsLocal()) {
      iter.NextLocal();
    }
    return iter;
  }

  const iterator LocalEnd() const { return End(); }

 protected:
  Client* client_ = nullptr;
  std::map<std::string, std::string> params_;

 private:
  size_t chunk_size_ = 0;
};

/**
 * @brief CollectionBuilder is the builder for collection objects.
 *
 * @tparam T The type of the underlying collection type, e.g.,
 *
 *       auto id = CollectionBuilder<Table>::Make(client, params);
 */
template <typename T>
class CollectionBuilder : public ObjectBuilder {
 public:
  static_assert(std::is_base_of<Object, T>::value,
                "Collection: not a vineyard object type");

  explicit CollectionBuilder(Client& client) : client_(client) {
    meta_.SetTypeName(type_name<collection_type_t<T>>());
    meta_.SetNBytes(0);
  }

  void AddMembers(const std::vector<ObjectID> objects,
                  const int64_t start_index = -1) {
    size_t starting = start_index < 0 ? chunk_size_ : start_index;
    for (size_t i = 0; i < objects.size(); ++i) {
      AddMember(starting + i, objects[i]);
    }
    chunk_size_ = std::max(chunk_size_, starting + objects.size());
  }

  void AddMembers(const std::vector<std::shared_ptr<Object>> objects,
                  const int64_t start_index = -1) {
    size_t starting = start_index < 0 ? chunk_size_ : start_index;
    for (size_t i = 0; i < objects.size(); ++i) {
      AddMember(starting + i, objects[i]);
    }
    chunk_size_ = std::max(chunk_size_, starting + objects.size());
  }

  void AddMember(const ObjectID& id) {
    meta_.AddMember(detail::index_to_key(chunk_size_++), id);
  }

  void AddMember(const ObjectMeta& meta) {
    meta_.AddMember(detail::index_to_key(chunk_size_++), meta);
  }

  void AddMember(const std::shared_ptr<Object>& object) {
    meta_.AddMember(detail::index_to_key(chunk_size_++), object);
  }

  Status AddMember(const std::shared_ptr<ObjectBuilder>& builder) {
    return this->AddMember(detail::index_to_key(chunk_size_++), builder);
  }

  void AddMember(const size_t index, const ObjectID& id) {
    meta_.AddMember(detail::index_to_key(index), id);
    chunk_size_ = std::max(chunk_size_, index + 1);
  }

  void AddMember(const size_t index, const ObjectMeta& meta) {
    meta_.AddMember(detail::index_to_key(index), meta);
    chunk_size_ = std::max(chunk_size_, index + 1);
  }

  void AddMember(const size_t index, const std::shared_ptr<Object>& object) {
    meta_.AddMember(detail::index_to_key(index), object);
    chunk_size_ = std::max(chunk_size_, index + 1);
  }

  Status AddMember(const size_t index,
                   const std::shared_ptr<ObjectBuilder>& builder) {
    std::shared_ptr<Object> object;
    RETURN_ON_ERROR(builder->Seal(client_, object));
    meta_.AddMember(detail::index_to_key(index), object);
    chunk_size_ = std::max(chunk_size_, index + 1);
    return Status::OK();
  }

  void AddMember(const std::string& key, const ObjectID& id) {
    meta_.AddMember(key, id);
    int64_t index = -1;
    if ((index = detail::index_from_key(key)) != -1) {
      chunk_size_ = std::max(chunk_size_, static_cast<size_t>(index) + 1);
    }
  }

  void AddMember(const std::string& key, const ObjectMeta& meta) {
    meta_.AddMember(key, meta);
    int64_t index = -1;
    if ((index = detail::index_from_key(key)) != -1) {
      chunk_size_ = std::max(chunk_size_, static_cast<size_t>(index) + 1);
    }
  }

  void AddMember(const std::string& key,
                 const std::shared_ptr<Object>& object) {
    meta_.AddMember(key, object);
    int64_t index = -1;
    if ((index = detail::index_from_key(key)) != -1) {
      chunk_size_ = std::max(chunk_size_, static_cast<size_t>(index) + 1);
    }
  }

  Status AddMember(const std::string& key,
                   const std::shared_ptr<ObjectBuilder>& builder) {
    std::shared_ptr<Object> object;
    RETURN_ON_ERROR(builder->Seal(client_, object));
    meta_.AddMember(key, object);
    int64_t index = -1;
    if ((index = detail::index_from_key(key)) != -1) {
      chunk_size_ = std::max(chunk_size_, static_cast<size_t>(index) + 1);
    }
    return Status::OK();
  }

  template <typename Value>
  void AddKeyValue(const std::string& key, const Value& value) {
    meta_.AddKeyValue(key, value);
  }

  void SetGlobal(const bool global = true) { meta_.SetGlobal(global); }

  Status Build(Client& client) override { return Status::OK(); }

  Status _Seal(Client& client, std::shared_ptr<Object>& object) override {
    ENSURE_NOT_SEALED(this);

    RETURN_ON_ERROR(this->Build(client));

    ObjectID id = InvalidObjectID();
    meta_.AddKeyValue("partitions_-size", chunk_size_);
    RETURN_ON_ERROR(client_.CreateMetaData(meta_, id));
    // mark the builder as sealed
    this->set_sealed(true);
    return client_.GetObject(id, object);
  }

 protected:
  Client& client_;

 private:
  ObjectMeta meta_;
  size_t chunk_size_ = 0;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_COLLECTION_H_
