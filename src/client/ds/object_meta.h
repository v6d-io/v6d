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

#ifndef SRC_CLIENT_DS_OBJECT_META_H_
#define SRC_CLIENT_DS_OBJECT_META_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "client/ds/core_types.h"
#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/typename.h"
#include "common/util/uuid.h"

namespace vineyard {

class Blob;
class Buffer;
class BufferSet;
class ClientBase;
class Client;
class Object;
class RPCClient;

/**
 * @brief ObjectMeta is the type for metadata of an Object. The ObjectMeta can
 * be treated as a *dict-like* type. If the metadata obtained
 * from vineyard, the metadata is readonly. Otherwise *key-value* attributes or
 * object members could be associated with the metadata to construct a new
 * vineyard object.
 */
class ObjectMeta {
 public:
  ObjectMeta();
  ~ObjectMeta();

  ObjectMeta(const ObjectMeta&);
  ObjectMeta& operator=(ObjectMeta const& other);

  /**
   * @brief Associate the client with the metadata.
   */
  void SetClient(ClientBase* client);

  /**
   * @brief Get the associate client with the metadata.
   */
  ClientBase* GetClient() const;

  /**
   * @brief Set the object ID for the metadata.
   */
  void SetId(const ObjectID& id);

  /**
   * @brief Get the corresponding object ID of the metadata.
   */
  const ObjectID GetId() const;

  /**
   * @brief Get the corresponding object signature of the metadata.
   */
  const Signature GetSignature() const;

  /**
   * @brief Reset the signatures in the metadata (for duplicating objects).
   */
  void ResetSignature();

  /**
   * @brief Mark the vineyard object as a global object.
   */
  void SetGlobal(bool global = true);

  /**
   * @brief Check whether the vineyard object is a global object.
   */
  const bool IsGlobal() const;

  /**
   * @brief Set the `typename` of the metadata. The `typename` will be used to
   * resolve in the ObjectFactory to create new object instances when get
   * objects from vineyard server.
   */
  void SetTypeName(const std::string& type_name);

  /**
   * @brief Get the `typename` of the metadata.
   */
  std::string const& GetTypeName() const;

  /**
   * @brief Set the `nbytes` attribute for the metadata, basically it indicates
   * the memory usage of the object.
   */
  void SetNBytes(const size_t nbytes);

  /**
   * @brief Get the `nbytes` attribute of the object. Note that the `nbytes`
   * attribute doesn't always reflect the TOTAL monopolistic space usage of the
   * bulk store, since two objects may share some blobs.
   */
  size_t const GetNBytes() const;

  /**
   * @brief Get the instance ID of vineyard server where the metadata is created
   * on.
   */
  InstanceID const GetInstanceId() const;

  /**
   * @brief Whether the object meta is the metadata of a local object.
   */
  bool const IsLocal() const;

  /**
   * @brief Mark the metadata as a "local" metadata to make sure the construct
   * process proceed.
   */
  void ForceLocal() const;

  /**
   * @brief Whether specific `key` exists in this metadata.
   */
  bool const Haskey(std::string const& key) const __attribute__((deprecated(
      "The method `Haskey()` is deprecated, use `HasKey()` instead.")));

  /**
   * @brief Whether specific `key` exists in this metadata.
   */
  bool const HasKey(std::string const& key) const;

  /**
   * @brief Reset the given key in the metadata.
   */
  void ResetKey(std::string const& key);

  /**
   * @brief Add a string value entry to the metadata.
   *
   * @param key The name of metadata entry.
   * @param value The value of the metadata entry.
   */
  void AddKeyValue(const std::string& key, const std::string& value);

  /**
   * @brief Add a generic value entry to the metadata.
   *
   * @param T The type of metadata's value.
   * @param key The name of metadata entry.
   * @param value The value of the metadata entry.
   */
  template <typename T>
  void AddKeyValue(const std::string& key, T const& value) {
    meta_[key] = value;
  }

  /**
   * @brief Add a generic set value entry to the metadata.
   *
   * @param T The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to JSON
   * array by `nlohmann::json`.
   * @param value The value of the metadata entry.
   */
  template <typename T>
  void AddKeyValue(const std::string& key, std::set<T> const& values) {
    meta_[key] = json_to_string(json(values));
  }

  /**
   * @brief Add a generic vector value entry to the metadata.
   *
   * @param T The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to JSON
   * array by `nlohmann::json`.
   * @param value The value of the metadata entry.
   */
  template <typename T>
  void AddKeyValue(const std::string& key, std::vector<T> const& values) {
    meta_[key] = json_to_string(json(values));
  }

  /**
   * @brief Add a generic vector value entry to the metadata.
   *
   * @param T The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to JSON
   * array by `nlohmann::json`.
   * @param value The value of the metadata entry.
   */
  template <typename T>
  void AddKeyValue(const std::string& key, Tuple<T> const& values) {
    meta_[key] = json_to_string(json(values));
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   std::map<std::string, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[kv.first] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   Map<std::string, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[kv.first] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   std::map<json, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[json_to_string(kv.first)] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key, Map<json, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[json_to_string(kv.first)] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   std::unordered_map<std::string, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[kv.first] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   UnorderedMap<std::string, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[kv.first] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   std::unordered_map<json, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[json_to_string(kv.first)] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a associated map value entry to the metadata.
   *
   * @param Value The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string.
   * @param value The value of the metadata entry.
   */
  template <typename Value>
  void AddKeyValue(const std::string& key,
                   UnorderedMap<json, Value> const& values) {
    json mapping;
    for (auto const& kv : values) {
      mapping[json_to_string(kv.first)] = kv.second;
    }
    AddKeyValue(key, mapping);
  }

  /**
   * @brief Add a `nlohmann::json` value entry to the metadata.
   *
   * @param T The type of metadata's value.
   * @param key The name of metadata entry, it will be first convert to string
   * by `nlohmann::json`.
   * @param value The value of the metadata entry.
   */
  void AddKeyValue(const std::string& key, json const& values);

  /**
   * @brief Get string metadata value.
   *
   * @param key The key of metadata.
   */
  const std::string GetKeyValue(const std::string& key) const {
    return meta_[key].get_ref<const std::string&>();
  }

  /**
   * @brief Get generic metadata value.
   *
   * @param T The type of metadata value.
   * @param key The key of metadata.
   */
  template <typename T>
  const T GetKeyValue(const std::string& key) const {
    return meta_[key].get<typename std::remove_cv<T>::type>();
  }

  /**
   * @brief Get generic metadata value, with automatically type deduction.
   *
   * @param T The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename T>
  void GetKeyValue(const std::string& key, T& value) const {
    value = meta_[key].get<typename std::remove_cv<T>::type>();
  }

  /**
   * @brief Get generic set metadata value, with automatically type deduction.
   *
   * @param T The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename T>
  void GetKeyValue(const std::string& key, std::set<T>& values) const {
    get_container(meta_, key, values);
  }

  /**
   * @brief Get generic vector metadata value, with automatically type
   * deduction.
   *
   * @param T The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename T>
  void GetKeyValue(const std::string& key, std::vector<T>& values) const {
    get_container(meta_, key, values);
  }

  /**
   * @brief Get generic vector metadata value, with automatically type
   * deduction.
   *
   * @param T The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename T>
  void GetKeyValue(const std::string& key, Tuple<T>& values) const {
    get_container(meta_, key, values);
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   std::map<std::string, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(kv.key(), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   Map<std::string, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(kv.key(), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   std::map<json, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(json::parse(kv.key()), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key, Map<json, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(json::parse(kv.key()), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   std::unordered_map<std::string, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(kv.key(), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   UnorderedMap<std::string, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(kv.key(), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   std::unordered_map<json, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(json::parse(kv.key()), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get associated map metadata value, with automatically type
   * deduction.
   *
   * @param Value The type of metadata value.
   * @param key The key of metadata.
   * @param value The result will be stored in `value`, the generic result is
   * pass by reference to help type deduction.
   */
  template <typename Value>
  void GetKeyValue(const std::string& key,
                   UnorderedMap<json, Value>& values) const {
    json tree;
    GetKeyValue(key, tree);
    for (auto const& kv : tree.items()) {
      values.emplace(json::parse(kv.key()), kv.value().get<Value>());
    }
  }

  /**
   * @brief Get json metadata value.
   *
   * @param key The key of metadata.
   * @param value The result will be stored in `value`.
   */
  void GetKeyValue(const std::string& key, json& value) const;

  /**
   * @brief Add member to ObjectMeta.
   *
   * @param name The name of member object.
   * @param member The metadata of member object to be added.
   */
  void AddMember(const std::string& name, const ObjectMeta& member);

  /**
   * @brief Add member to ObjectMeta.
   *
   * @param name The name of member object.
   * @param member The member object to be added.
   */

  void AddMember(const std::string& name, const Object& member);

  /**
   * @brief Add member to ObjectMeta.
   *
   * @param name The name of member object.
   * @param member The member object to be added.
   */
  void AddMember(const std::string& name, const Object* member);

  /**
   * @brief Add member to ObjectMeta.
   *
   * @param name The name of member object.
   * @param member The member object to be added.
   */
  void AddMember(const std::string& name,
                 const std::shared_ptr<Object>& member);

  /**
   * @brief Add member to ObjectMeta.
   *
   * @param name The name of member object.
   * @param member The object ID of member object to be added.
   */
  void AddMember(const std::string& name, const ObjectID member_id);

  /**
   * @brief Get member value from vineyard.
   *
   * @param name The name of member object.
   *
   * @return member The member object.
   */
  std::shared_ptr<Object> GetMember(const std::string& name) const;

  /**
   * @brief Get member value from vineyard.
   *
   * @param name The name of member object.
   *
   * @return member The member object.
   */
  Status GetMember(const std::string& name,
                   std::shared_ptr<Object>& object) const;

  /**
   * @brief Get member value from vineyard.
   *
   * @param name The name of member object.
   *
   * @return member The member object.
   */
  template <typename T>
  std::shared_ptr<T> GetMember(const std::string& name) const {
    return std::dynamic_pointer_cast<T>(GetMember(name));
  }

  /**
   * @brief Get member value from vineyard.
   *
   * @param name The name of member object.
   *
   * @return member The member object.
   */
  template <typename T>
  Status GetMember(const std::string& name, std::shared_ptr<T>& object) const {
    std::shared_ptr<Object> _object;
    RETURN_ON_ERROR(GetMember(name, _object));
    object = std::dynamic_pointer_cast<T>(_object);
    if (object == nullptr) {
      return Status::ObjectTypeError(type_name<T>(),
                                     GetMemberMeta(name).GetTypeName());
    } else {
      return Status::OK();
    }
  }

  /**
   * @brief Get member's ObjectMeta value.
   *
   * @param name The name of member object.
   *
   * @return member The metadata of member object. will be stored in `value`.
   */
  ObjectMeta GetMemberMeta(const std::string& name) const;

  /**
   * @brief Get member's ObjectMeta value.
   *
   * @param name The name of member object.
   * @param meta The metadata of member object.
   *
   * @return Whether the member metadata has been found.
   */
  Status GetMemberMeta(const std::string& name, ObjectMeta& meta) const;

  /**
   * @brief Get buffer member (directed or indirected) from the metadata. The
   * metadata should has already been initialized.
   */
  Status GetBuffer(const ObjectID blob_id,
                   std::shared_ptr<Buffer>& buffer) const;

  void SetBuffer(const ObjectID& id, const std::shared_ptr<Buffer>& buffer);

  /**
   * @brief Reset the metadata as an initialized one.
   */
  void Reset();

  /**
   * @brief Compute the memory usage of this object.
   */
  size_t MemoryUsage() const;

  /**
   * @brief Compute the memory usage of this object, in a json tree format.
   */
  size_t MemoryUsage(json& usages, const bool pretty = true) const;

  /**
   * @brief Get the associate timestamp (in milliseconds) inside the metadata, 0
   *        means unknown.
   */
  uint64_t Timestamp() const;

  json Labels() const;

  const std::string Label(const std::string& key) const;

  std::string ToString() const;

  void PrintMeta() const;

  const bool incomplete() const;

  // FIXME: the following three methods should be `protected`
  const json& MetaData() const;

  json& MutMetaData();

  void SetMetaData(ClientBase* client, const json& meta);

  /**
   * Construct object metadata from unsafe sources.
   */
  static std::unique_ptr<ObjectMeta> Unsafe(std::string meta, size_t nobjects,
                                            ObjectID* objects,
                                            uintptr_t* pointers, size_t* sizes);

  /**
   * Construct object metadata from unsafe sources.
   */
  static std::unique_ptr<ObjectMeta> Unsafe(json meta, size_t nobjects,
                                            ObjectID* objects,
                                            uintptr_t* pointers, size_t* sizes);

  using const_iterator =
      nlohmann::detail::iteration_proxy_value<json::const_iterator>;
  const_iterator begin() const { return meta_.items().begin(); }
  const_iterator end() const { return meta_.items().end(); }

  const std::shared_ptr<BufferSet>& GetBufferSet() const;

 private:
  void SetInstanceId(const InstanceID instance_id);

  void SetSignature(const Signature signature);

  // hold a client_ reference, since we already hold blobs in metadata, which,
  // depends on that the "client_" should be valid.
  ClientBase* client_ = nullptr;
  json meta_;

  // associated blobs
  mutable std::shared_ptr<BufferSet> buffer_set_ = nullptr;

  // incomplete: whether the metadata has incomplete member, introduced by
  // `AddMember(name, member_id)`.
  bool incomplete_ = false;

  // force local: make it as a local metadata even when no client associated.
  mutable bool force_local_ = false;

  friend class ClientBase;
  friend class Client;
  friend class PlasmaClient;
  friend class RPCClient;

  friend class Blob;
  friend class RemoteBlob;
  friend class BlobWriter;
};

template <>
const json ObjectMeta::GetKeyValue<json>(const std::string& key) const;

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_OBJECT_META_H_
