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

#ifndef SRC_CLIENT_DS_BLOB_H_
#define SRC_CLIENT_DS_BLOB_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "arrow/buffer.h"

#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

class BlobWriter;
class BlobSet;
class Client;
class ObjectMeta;

/**
 * @brief The unit to store data payload in vineyard.
 * When the client gets a blob from vineyard, the vineyard server maps
 * a chunk of memory from its memory space to the client space in a
 * zero-copy fashion.
 */
class Blob : public Registered<Blob> {
 public:
  /**
   * @brief Get the size of the blob, i.e., the number of bytes of the data
   * payload in the blob.
   *
   * @return The size of the blob.
   */
  size_t size() const;

  /**
   * @brief Get the const data pointer of the data payload in the blob.
   *
   * @return The const data pointer.
   */
  const char* data() const;

  /**
   * @brief Get the arrow buffer of the blob.
   *
   * @return The arrow buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<arrow::Buffer>& Buffer() const;

  static std::shared_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(std::shared_ptr<Blob>{new Blob()});
  }

  /**
   * @brief Construct the blob locally for the given object meta.
   *
   * @param meta The given object meta.
   */
  void Construct(ObjectMeta const& meta) override;

  /**
   * @brief Dump the buffer for debugging.
   */
  void Dump() const;

  /**
   * @brief Create an empty blob in the vineyard server.
   *
   * @param client The client connected to the vineyard server.
   */
  static std::shared_ptr<Blob> MakeEmpty(Client& client);

 private:
  /** The default constructor is only used in BlobWriter.
   */
  Blob();
  Blob(const ObjectID id, const size_t size);
  Blob(const ObjectID id, const size_t size,
       std::shared_ptr<arrow::Buffer> const& buffer);

  const std::shared_ptr<arrow::Buffer>& BufferUnsafe() const;

  size_t size_;
  std::shared_ptr<arrow::Buffer> buffer_;

  friend class Client;
  friend class RPCClient;
  friend class BlobWriter;
  friend class BlobSet;
  friend class ObjectMeta;
};

/**
 * @brief The writer to write a blob in vineyard.
 * The writer is initialized in the client with a local buffer and its size,
 * and a blob in vineyard will be created when Build is invoked.
 */
class BlobWriter : public ObjectBuilder {
 public:
  /**
   * @brief Get the size of the blob, i.e., the number of bytes of the data
   * payload in the blob.
   *
   * @return The size of the blob.
   */
  size_t size() const;

  /**
   * @brief Get the data pointer of the data payload in the blob.
   *
   * @return The data pointer.
   */
  char* data();

  /**
   * @brief Get the const data pointer of the data payload in the blob.
   *
   * @return The const data pointer.
   */
  const char* data() const;

  /**
   * @brief Get the mutable buffer of the blob.
   *
   * @return The mutable buffer of the blob, which can be modified
   * to update the content in the blob.
   */
  const std::shared_ptr<arrow::MutableBuffer>& Buffer() const;

  /**
   * @brief Build a blob in vineyard server.
   *
   * @param client The client connected to the vineyard server.
   */
  Status Build(Client& client) override;

  /**
   * @brief Add key-value metadata for the blob.
   *
   * @param key The key of the metadata.
   * @param value The value of the metadata.
   */
  void AddKeyValue(std::string const& key, std::string const& value);

  /**
   * @brief Add key-value metadata for the blob.
   *
   * @param key The key of the metadata.
   * @param value The value of the metadata.
   */
  void AddKeyValue(std::string const& key, std::string&& value);

  /**
   * @brief Dump the buffer for debugging.
   */
  void Dump() const;

 protected:
  std::shared_ptr<Object> _Seal(Client& client) override;

 private:
  BlobWriter(ObjectID const object_id,
             std::shared_ptr<arrow::MutableBuffer> const& buffer)
      : object_id_(object_id), buffer_(buffer) {}

  ObjectID object_id_;
  std::shared_ptr<arrow::MutableBuffer> buffer_;
  // Allowing blobs have extra key-value metadata
  std::unordered_map<std::string, std::string> metadata_;

  friend class Client;
  friend class RPCClient;
};

/**
 * @brief A set of blobs that been associated with an object and its member
 */
class BlobSet {
 public:
  const std::unordered_set<ObjectID>& AllBlobIds() const { return ids_; }

  const std::unordered_map<ObjectID, Blob>& AllBlobs() const { return blobs_; }

  void EmplaceId(ObjectID const id, size_t const size, bool local);

  void EmplaceBlob(ObjectID const id,
                   std::shared_ptr<arrow::Buffer> const& buffer);

  void Extend(BlobSet const& others);

  void Extend(std::shared_ptr<BlobSet> const& others);

  bool Contains(ObjectID const id) const;

 private:
  // fetchable blob ids set, i.e., local blobs
  std::unordered_set<ObjectID> ids_;
  // blob ids to blobs mapping: local blobs + remote blobs
  std::unordered_map<ObjectID, Blob> blobs_;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_BLOB_H_
