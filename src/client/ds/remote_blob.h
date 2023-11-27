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

#ifndef SRC_CLIENT_DS_REMOTE_BLOB_H_
#define SRC_CLIENT_DS_REMOTE_BLOB_H_

#include <cstdint>
#include <limits>
#include <memory>

#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace arrow {
class Buffer;
}  // namespace arrow

namespace vineyard {

class Buffer;
class MutableBuffer;
class ObjectMeta;
class RemoteBlobWriter;
class RPCClient;

/**
 * @brief The unit to store data payload in vineyard.
 *
 * When the client gets a blob from vineyard, the vineyard server maps
 * a chunk of memory from its memory space to the client space in a
 * zero-copy fashion.
 */
class RemoteBlob : public Registered<RemoteBlob> {
 public:
  ObjectID id() const;

  InstanceID instance_id() const;

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
   * @brief Get the buffer of the blob.
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<vineyard::Buffer>& Buffer() const;

  /**
   * @brief Construct the blob locally for the given object meta.
   *
   * @param meta The given object meta.
   */
  void Construct(ObjectMeta const& meta) override;

  /**
   * @brief Get the arrow buffer of the blob.
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<arrow::Buffer> ArrowBuffer() const;

  /**
   * @brief Get the buffer of the blob, ensure a valid shared_ptr been
   * returned even the blob is empty (size == 0).
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<vineyard::Buffer> BufferOrEmpty() const;

  /**
   * @brief Get the arrow buffer of the blob, ensure a valid shared_ptr been
   * returned even the blob is empty (size == 0).
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<arrow::Buffer> ArrowBufferOrEmpty() const;

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<RemoteBlob>{new RemoteBlob()});
  }

  /**
   * @brief Dump the buffer for debugging.
   */
  void Dump() const;

 private:
  /**
   * @brief Construct an empty RemoteBlob
   */
  RemoteBlob() {
    this->id_ = InvalidObjectID();
    this->size_ = std::numeric_limits<size_t>::max();
    this->buffer_ = nullptr;
  }

  RemoteBlob(const ObjectID id, const InstanceID instance_id,
             const size_t size);

  char* mutable_data() const;

  ObjectID id_;
  InstanceID instance_id_;
  size_t size_ = 0;
  std::shared_ptr<vineyard::Buffer> buffer_ = nullptr;

  friend class RPCClient;
  friend class RemoteBlobWriter;
  friend class ObjectMeta;
};

/**
 * @brief The writer to write a blob in vineyard.
 *
 * The writer is initialized in the client with a local buffer and its size,
 * and a blob in vineyard will be created when Build is invoked.
 */
class RemoteBlobWriter {
 public:
  explicit RemoteBlobWriter(const size_t size);
  explicit RemoteBlobWriter(std::shared_ptr<MutableBuffer> const& buffer);

  ~RemoteBlobWriter();

  static std::shared_ptr<RemoteBlobWriter> Make(const size_t size);

  static std::shared_ptr<RemoteBlobWriter> Wrap(const uint8_t* data,
                                                const size_t size);

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
  const std::shared_ptr<MutableBuffer>& Buffer() const;

  /**
   * @brief Abort the blob builder.
   *
   * @param client Release the blob builder object if it is not sealed.
   */
  Status Abort();

  /**
   * @brief Dump the buffer for debugging.
   */
  void Dump() const;

 private:
  std::shared_ptr<MutableBuffer> buffer_;

  friend class RPCClient;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_REMOTE_BLOB_H_
