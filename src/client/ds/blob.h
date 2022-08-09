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

#ifndef SRC_CLIENT_DS_BLOB_H_
#define SRC_CLIENT_DS_BLOB_H_

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/ds/i_object.h"
#include "common/memory/payload.h"
#include "common/util/uuid.h"

namespace vineyard {

class BlobWriter;
class BufferSet;
class Client;
class PlasmaClient;
class ObjectMeta;

/**
 * @brief The unit to store data payload in vineyard.
 *
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
   * Note that the size of blob is the "allocated size" of the blob, and may
   * (usually) not be the same value of the requested size.
   *
   * @return The (allocated) size of the blob.
   */
  size_t size() const __attribute__((
      deprecated("The method `size()` no longer report accurate requested "
                 "memory size, use allocated_size() instead.")));

  /**
   * @brief Get the allocated size of the blob, i.e., the number of bytes of the
   * data payload in the blob.
   *
   * @return The allocated size of the blob.
   */
  size_t allocated_size() const;

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

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(std::unique_ptr<Blob>{new Blob()});
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

  /**
   * @brief Create the blob from a buffer from the client-side allocator.
   *
   * @param object_id The object ID of this blob.
   * @param pointer The address of buffer in the client-side allocator.
   * @param size The estimated size of the buffer.
   */
  static std::shared_ptr<Blob> FromAllocator(Client& client,
                                             const ObjectID object_id,
                                             const uintptr_t pointer,
                                             const size_t size);

  /**
   * @brief Create the blob from a given buffer. If the buffer already lies in
   * the vineyardd, it would return immediately without copying, otherwise a
   *        blob writer will be created and the content of the buffer will be
   *        copied into.
   *
   * @param pointer The address of the buffer.
   * @param size The estimated size of the buffer.
   */
  static std::shared_ptr<Blob> FromPointer(Client& client,
                                           const uintptr_t pointer,
                                           const size_t size);

 private:
  /**
   * The default constructor is only used in BlobWriter.
   */
  Blob() {
    this->id_ = InvalidObjectID();
    this->size_ = std::numeric_limits<size_t>::max();
    this->buffer_ = nullptr;
  }

  const std::shared_ptr<arrow::Buffer>& BufferUnsafe() const;

  size_t size_ = 0;
  std::shared_ptr<arrow::Buffer> buffer_ = nullptr;

  friend class Client;
  friend class PlasmaClient;
  friend class RPCClient;
  friend class BlobWriter;
  friend class BufferSet;
  friend class ObjectMeta;
};

/**
 * @brief The writer to write a blob in vineyard.
 *
 * The writer is initialized in the client with a local buffer and its size,
 * and a blob in vineyard will be created when Build is invoked.
 */
class BlobWriter : public ObjectBuilder {
 public:
  /**
   * @brief Return the object id of this blob builder. Note that before sealing
   * the blob builder the object id cannot be used to get "Blob" objects.
   *
   * @return The ObjectID of the blob writer.
   */
  ObjectID id() const;

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
   * @brief Abort the blob builder.
   *
   * @param client Release the blob builder object if it is not sealed.
   */
  Status Abort(Client& client);

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
  BlobWriter(ObjectID const object_id, const Payload& payload,
             std::shared_ptr<arrow::MutableBuffer> const& buffer)
      : object_id_(object_id), payload_(payload), buffer_(buffer) {}

  BlobWriter(ObjectID const object_id, Payload&& payload,
             std::shared_ptr<arrow::MutableBuffer> const& buffer)
      : object_id_(object_id), payload_(payload), buffer_(buffer) {}

  ObjectID object_id_;
  Payload payload_;
  std::shared_ptr<arrow::MutableBuffer> buffer_;
  // Allowing blobs have extra key-value metadata
  std::unordered_map<std::string, std::string> metadata_;

  friend class Client;
  friend class PlasmaClient;
  friend class RPCClient;
};

/**
 * @brief A set of (readonly) buffers that been associated with an object and
 * its members (recursively).
 */
class BufferSet {
 public:
  const std::set<ObjectID>& AllBufferIds() const { return buffer_ids_; }

  const std::map<ObjectID, std::shared_ptr<arrow::Buffer>>& AllBuffers() const {
    return buffers_;
  }

  Status EmplaceBuffer(ObjectID const id);

  Status EmplaceBuffer(ObjectID const id,
                       std::shared_ptr<arrow::Buffer> const& buffer);

  void Extend(BufferSet const& others);

  void Extend(std::shared_ptr<BufferSet> const& others);

  bool Contains(ObjectID const id) const;

  bool Get(ObjectID const id, std::shared_ptr<arrow::Buffer>& buffer) const;

 private:
  // blob ids to buffer mapping: local blobs (not null) + remote blobs (null).
  std::set<ObjectID> buffer_ids_;
  std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers_;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_BLOB_H_
