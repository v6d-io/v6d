/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#ifndef SRC_CLIENT_CLIENT_H_
#define SRC_CLIENT_CLIENT_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/client_base.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "common/memory/payload.h"
#include "common/util/lifecycle.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class Blob;
class BlobWriter;

namespace detail {

/**
 * @brief MmapEntry represents a memory-mapped fd on the client side. The fd
 * can be mmapped as readonly or readwrite memory.
 */
class MmapEntry {
 public:
  MmapEntry(int fd, int64_t map_size, bool readonly, bool realign = false);

  ~MmapEntry();

  /**
   * @brief Map the shared memory represents by `fd_` as readonly memory.
   *
   * @returns A untyped pointer that points to the shared readonly memory.
   */
  uint8_t* map_readonly();

  /**
   * @brief Map the shared memory represents by `fd_` as writeable memory.
   *
   * @returns A untyped pointer that points to the shared writeable memory.
   */
  uint8_t* map_readwrite();

  int fd() { return fd_; }

 private:
  /// The associated file descriptor on the client.
  int fd_;
  /// The result of mmap for this file descriptor.
  uint8_t *ro_pointer_, *rw_pointer_;
  /// The length of the memory-mapped file.
  size_t length_;
};

class SharedMemoryManager {
 public:
  explicit SharedMemoryManager(int vineyard_conn);

  Status Mmap(int fd, int64_t map_size, bool readonly, bool realign,
              uint8_t** ptr);

  bool Exists(const uintptr_t target);

  bool Exists(const void* target);

 private:
  // UNIX-domain socket
  int vineyard_conn_ = -1;

  // mmap table
  std::unordered_map<int, std::unique_ptr<MmapEntry>> mmap_table_;

  // sorted shm segments for fast "if exists" query
  std::set<std::pair<uintptr_t, size_t>> segments_;
};

}  // namespace detail

// Track the reference count in client-side to support object RAII.
template <typename ID, typename P, typename Der>
class UsageTracker : public LifeCycleTracker<ID, P, UsageTracker<ID, P, Der>> {
 public:
  UsageTracker() {}

  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t change) {
    auto elem = object_in_use_.find(id);
    if (elem != object_in_use_.end()) {
      elem->second->ref_cnt += change;
      ref_cnt = elem->second->ref_cnt;
      return Status::OK();
    }
    return Status::ObjectNotExists();
  }

  Status FetchOnLocal(ID const& id, P& payload) {
    auto elem = object_in_use_.find(id);
    if (elem != object_in_use_.end()) {
      payload = *(elem->second);
      if (payload.IsSealed()) {
        return Status::OK();
      } else {
        return Status::ObjectNotSealed();
      }
    }
    return Status::ObjectNotExists();
  }

  Status SealUsage(ID const& id) {
    auto elem = object_in_use_.find(id);
    if (elem != object_in_use_.end()) {
      elem->second->is_sealed = true;
      return Status::OK();
    }
    return Status::ObjectNotExists();
  }

  Status AddUsage(ID const& id, P const& payload) {
    auto elem = object_in_use_.find(id);
    if (elem == object_in_use_.end()) {
      object_in_use_[id] = std::make_shared<P>(payload);
      object_in_use_[id]->ref_cnt = 0;
    }
    return this->IncreaseReferenceCount(id);
  }

  Status RemoveUsage(ID const& id) { return this->DecreaseReferenceCount(id); }

  Status OnRelease(ID const& id) { return this->Self().OnRelease(id); }

  Status OnDelete(ID const& id) { return Self().OnDelete(id); }

 private:
  inline Der& Self() { return static_cast<Der&>(*this); }
  // Track the objects' usage.
  std::unordered_map<ID, std::shared_ptr<P>> object_in_use_;
};

class BasicIPCClient : public ClientBase {
 public:
  BasicIPCClient();

  ~BasicIPCClient() {}
  /**
   * @brief Connect to vineyardd using the given UNIX domain socket
   * `ipc_socket` with the given store type.
   *
   * @param ipc_socket Location of the UNIX domain socket.
   * @param bulk_store_type The name of the bulk store.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& ipc_socket,
                 std::string const& bulk_store_type);

  /**
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   * @param bulk_store_type The name of the bulk store.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket,
              std::string const& bulk_store_type);

 protected:
  std::shared_ptr<detail::SharedMemoryManager> shm_;
};

class Client;
class PlasmaClient;

/**
 * @brief Vineyard's IPC Client connects to to UNIX domain socket of the
 *        vineyard server. Vineyard's IPC Client talks to vineyard server
 *        and manipulate objects in vineyard.
 */
class Client : public BasicIPCClient {
 public:
  Client() {}

  ~Client() override;

  /**
   * @brief Connect to vineyard using the UNIX domain socket file specified by
   *        the environment variable `VINEYARD_IPC_SOCKET`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect();

  /**
   * @brief Connect to vineyardd using the given UNIX domain socket
   * `ipc_socket`.
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& ipc_socket);

  /**
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket);

  /**
   * @brief Create a new client using self UNIX domain socket.
   */
  Status Fork(Client& client);

  /**
   * @brief Get a default client reference, using the UNIX domain socket file
   *        specified by the environment variable `VINEYARD_IPC_SOCKET`.
   *
   * @return A reference of the default Client instance.
   */
  static Client& Default();

  /**
   * @brief Obtain metadata from vineyard server.
   *
   * @param id The object id to get.
   * @param meta_data The result metadata will be store in `meta_data` as return
   * value.
   * @param sync_remote Whether to trigger an immediate remote metadata
   *        synchronization before get specific metadata. Default is false.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetMetaData(const ObjectID id, ObjectMeta& meta_data,
                     const bool sync_remote = false) override;

  /**
   * @brief Obtain multiple metadatas from vineyard server.
   *
   * @param ids The object ids to get.
   * @param meta_data The result metadata will be store in `meta_data` as return
   * value.
   * @param sync_remote Whether to trigger an immediate remote metadata
   *        synchronization before get specific metadata. Default is false.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetMetaData(const std::vector<ObjectID>& id, std::vector<ObjectMeta>&,
                     const bool sync_remote = false);

  /**
   * @brief Create a blob in vineyard server. When creating a blob, vineyard
   * server's bulk allocator will prepare a block of memory of the requested
   * size, the map the memory to client's process to share the allocated memory.
   *
   * @param size The size of requested blob.
   * @param blob The result mutable blob will be set in `blob`.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status CreateBlob(size_t size, std::unique_ptr<BlobWriter>& blob);

  /**
   * @brief Allocate a chunk of given size in vineyard for a stream. When the
   * request cannot be statisfied immediately, e.g., vineyard doesn't have
   * enough memory or the specified has accumulated too many chunks, the request
   * will be blocked until the request been processed.
   *
   * @param id The id of the stream.
   * @param size The size of the chunk to allocate.
   * @param blob The allocated mutable buffer will be set in `blob`.
   *
   * @return Status that indicates whether the allocation has succeeded.
   */
  Status GetNextStreamChunk(ObjectID const id, size_t const size,
                            std::unique_ptr<arrow::MutableBuffer>& blob);

  // bring the overloadings in parent class to current scope.
  using ClientBase::PullNextStreamChunk;

  /**
   * @brief Pull a chunk from a stream. When there's no more chunk available in
   * the stream, i.e., the stream has been stoped, a status code
   * `kStreamDrained` or `kStreamFinish` will be returned, otherwise the reader
   * will be blocked until writer creates a new chunk in the stream.
   *
   * @param id The id of the stream.
   * @param chunk The immutable chunk generated by the writer of the stream.
   *
   * @return Status that indicates whether the polling has succeeded.
   */
  Status PullNextStreamChunk(ObjectID const id,
                             std::unique_ptr<arrow::Buffer>& chunk);

  /**
   * @brief Get an object from vineyard. The ObjectFactory will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   *
   * @return A std::shared_ptr<Object> that can be safely cast to the underlying
   * concrete object type. When the object doesn't exists an std::runtime_error
   * exception will be raised.
   */
  std::shared_ptr<Object> GetObject(const ObjectID id);

  /**
   * @brief Get an object from vineyard. The ObjectFactory will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   * @param object The result object will be set in parameter `object`.
   *
   * @return When errors occur during the request, this method won't throw
   * exceptions, rather, it results a status to represents the error.
   */
  Status GetObject(const ObjectID id, std::shared_ptr<Object>& object);

  /**
   * @brief Get an object from vineyard. The type parameter `T` will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   *
   * @return A std::shared_ptr<Object> that can be safely cast to the underlying
   * concrete object type. When the object doesn't exists an std::runtime_error
   * exception will be raised.
   */
  template <typename T>
  std::shared_ptr<T> GetObject(const ObjectID id) {
    return std::dynamic_pointer_cast<T>(GetObject(id));
  }

  /**
   * @brief Get an object from vineyard. The type parameter `T` will be used to
   * resolve the constructor of the object.
   *
   * This method can be used to get concrete object from vineyard without
   * explicitly `dynamic_cast`, and the template type parameter can be deduced
   * in many situations:
   *
   * \code{.cpp}
   *    std::shared_ptr<Array<int>> int_array;
   *    client.GetObject(id, int_array);
   * \endcode
   *
   * @param id The object id to get.
   * @param object The result object will be set in parameter `object`.
   *
   * @return When errors occur during the request, this method won't throw
   * exceptions, rather, it results a status to represents the error.
   */
  template <typename T>
  Status GetObject(const ObjectID id, std::shared_ptr<T>& object) {
    std::shared_ptr<Object> _object;
    RETURN_ON_ERROR(GetObject(id, _object));
    object = std::dynamic_pointer_cast<T>(_object);
    if (object == nullptr) {
      return Status::ObjectNotExists("object not exists: " +
                                     ObjectIDToString(id));
    } else {
      return Status::OK();
    }
  }

  /**
   * @brief Get multiple objects from vineayrd.
   *
   * @param ids The object IDs to get.
   *
   * @return A list of objects.
   */
  std::vector<std::shared_ptr<Object>> GetObjects(
      const std::vector<ObjectID>& ids);

  /**
   * @brief List object metadatas in vineyard, using the given typename
   * patterns.
   *
   * @param pattern The pattern string that will be used to matched against
   * objects' `typename`.
   * @param regex Whether the pattern is a regular expression pattern. Default
   * is false. When `regex` is false, the pattern will be treated as a glob
   * pattern.
   * @param limit The number limit for how many objects will be returned at
   * most.
   *
   * @return A vector of object metadatas that listed from vineyard server.
   */
  std::vector<ObjectMeta> ListObjectMeta(std::string const& pattern,
                                         const bool regex = false,
                                         size_t const limit = 5,
                                         bool nobuffer = false);

  /**
   * @brief List objects in vineyard, using the given typename patterns.
   *
   * @param pattern The pattern string that will be used to matched against
   * objects' `typename`.
   * @param regex Whether the pattern is a regular expression pattern. Default
   * is false. When `regex` is false, the pattern will be treated as a glob
   * pattern.
   * @param limit The number limit for how many objects will be returned at
   * most.
   *
   * @return A vector of objects that listed from vineyard server.
   */
  std::vector<std::shared_ptr<Object>> ListObjects(std::string const& pattern,
                                                   const bool regex = false,
                                                   size_t const limit = 5);

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const void* target) const;

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const uintptr_t target) const;

  /**
   * Get the allocated size for the given object.
   */
  Status AllocatedSize(const ObjectID id, size_t& size);

  Status CreateArena(const size_t size, int& fd, size_t& available_size,
                     uintptr_t& base, uintptr_t& space);

  Status ReleaseArena(const int fd, std::vector<size_t> const& offsets,
                      std::vector<size_t> const& sizes);

  using BasicIPCClient::ShallowCopy;
  /**
   * @brief Move the selected objects from the source session to the target
   */
  Status ShallowCopy(ObjectID const id, ObjectID& target_id,
                     Client& source_client);

  Status ShallowCopy(PlasmaID const plasma_id, ObjectID& target_id,
                     PlasmaClient& source_client);

 protected:
  Status CreateBuffer(const size_t size, ObjectID& id, Payload& payload,
                      std::shared_ptr<arrow::MutableBuffer>& buffer);

  /**
   * @brief Get a blob from vineyard server. When obtaining blobs from vineyard
   * server, the memory address in the server process will be mmapped to the
   * client's process to share the memory.
   *
   * @param id Object id for the blob to get.
   * @param buffer: The result immutable blob will be set in `blob`. Note that
   * blob is special, since it can be get as immutable object before sealing.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status GetBuffer(const ObjectID id, std::shared_ptr<arrow::Buffer>& buffer);

  /**
   * @brief Get a set of blobs from vineyard server. See also `GetBuffer`.
   *
   * @param ids Object ids for the blobs to get.
   * @param buffers: The result immutable blobs will be added to `buffers`.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status GetBuffers(
      const std::set<ObjectID>& ids,
      std::map<ObjectID, std::shared_ptr<arrow::Buffer>>& buffers);

  Status GetBufferSizes(const std::set<ObjectID>& ids,
                        std::map<ObjectID, size_t>& sizes);

  /**
   * @brief An (unsafe) internal-usage method that drop the buffer, without
   * checking the dependency. To achieve the "DeleteObject" semantic for blobs,
   * use `DelData` instead.
   *
   * Note that the target buffer could be both sealed and unsealed.
   */
  Status DropBuffer(const ObjectID id, const int fd);

  Status Seal(ObjectID const& object_id);

 private:
  friend class Blob;
  friend class BlobWriter;
};

class PlasmaClient
    : public BasicIPCClient,
      public UsageTracker<PlasmaID, PlasmaPayload, PlasmaClient> {
 public:
  PlasmaClient() {}

  ~PlasmaClient() override;

  Status GetMetaData(const ObjectID id, ObjectMeta& meta_data,
                     const bool sync_remote = false) override;

  /**
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket);

  /**
   * @brief Connect to vineyardd using the given UNIX domain socket
   * `ipc_socket`.
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& ipc_socket);

  /**
   * @brief Create a blob in vineyard server. When creating a blob, vineyard
   * server's bulk allocator will prepare a block of memory of the requested
   * size, the map the memory to client's process to share the allocated memory.
   *
   * @param plasma_id The id of plasma data.
   * @param size The size of requested blob.
   * @param plasma_size The size of plasma data.
   * @param blob The result mutable blob will be set in `blob`.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status CreateBuffer(PlasmaID plasma_id, size_t size, size_t plasma_size,
                      std::unique_ptr<BlobWriter>& blob);

  Status GetPayloads(std::set<PlasmaID> const& plasma_ids,
                     std::map<PlasmaID, PlasmaPayload>& plasma_payloads);

  /**
   * Used only for integration.
   */
  Status GetBuffers(
      std::set<PlasmaID> const& plasma_ids,
      std::map<PlasmaID, std::shared_ptr<arrow::Buffer>>& buffers);

  Status ShallowCopy(PlasmaID const plasma_id, PlasmaID& target_pid,
                     PlasmaClient& source_client);

  Status ShallowCopy(ObjectID const id, std::set<PlasmaID>& target_pids,
                     Client& source_client);

  Status Seal(PlasmaID const& object_id);

  Status Release(PlasmaID const& id);

  Status Delete(PlasmaID const& id);

  /// For UsageTracker only
  Status OnFetch(PlasmaID const& id,
                 std::shared_ptr<PlasmaPayload> const& plasma_payload);

  /// For UsageTracker only
  Status OnRelease(PlasmaID const& id);

  /// For UsageTracker only
  Status OnDelete(PlasmaID const& id);

 private:
  using ClientBase::Release;  // avoid warning
};

}  // namespace vineyard

#endif  // SRC_CLIENT_CLIENT_H_
