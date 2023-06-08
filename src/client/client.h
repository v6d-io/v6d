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

#ifndef SRC_CLIENT_CLIENT_H_
#define SRC_CLIENT_CLIENT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "client/client_base.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "common/memory/gpu/unified_memory.h"
#include "common/memory/payload.h"
#include "common/util/lifecycle.h"
#include "common/util/protocols.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class Blob;
class BlobWriter;
class Buffer;
class MutableBuffer;

namespace detail {

class SharedMemoryManager;

/**
 * @brief MmapEntry represents a memory-mapped fd on the client side. The fd
 * can be mmapped as readonly or readwrite memory.
 */
class MmapEntry {
 public:
  MmapEntry(int fd, int64_t map_size, uint8_t* pointer, bool readonly,
            bool realign = false);

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
  /// The pointer at the server side, for obtaining the object id by given
  /// offset.
  uint8_t* pointer;
  /// The result of mmap for this file descriptor.
  uint8_t *ro_pointer_, *rw_pointer_;
  /// The length of the memory-mapped file.
  size_t length_;

  friend class SharedMemoryManager;
};

class SharedMemoryManager {
 public:
  explicit SharedMemoryManager(int vineyard_conn);

  Status Mmap(int fd, int64_t map_size, uint8_t* pointer, bool readonly,
              bool realign, uint8_t** ptr);

  Status Mmap(int fd, ObjectID id, int64_t map_size, size_t data_size,
              size_t data_offset, uint8_t* pointer, bool readonly, bool realign,
              uint8_t** ptr);

  // compute if the given fd requireds a recv_fd and mmap
  int PreMmap(int fd);

  // compute the set of fds that needs to `recv` from the server
  void PreMmap(int fd, std::vector<int>& fds, std::set<int>& dedup);

  bool Exists(const uintptr_t target)
      __attribute__((deprecated("Use Exists(target, object_id) instead.")));

  bool Exists(const void* target)
      __attribute__((deprecated("Use Exists(target, object_id) instead.")));

  bool Exists(const uintptr_t target, ObjectID& object_id);

  bool Exists(const void* target, ObjectID& object_id);

 private:
  ObjectID resolveObjectID(const uintptr_t target, const uintptr_t key,
                           const uintptr_t data_size, const ObjectID object_id);

  // UNIX-domain socket
  int vineyard_conn_ = -1;

  // mmap table
  std::unordered_map<int, std::unique_ptr<MmapEntry>> mmap_table_;

  // sorted shm segments for fast "if exists" query
  std::map<uintptr_t, std::pair<size_t, ObjectID>> segments_;
};

/**
 * @brief UsageTracker is a CRTP class optimize the LifeCycleTracker by caching
 * the reference count and payload on client to avoid frequent IPCs like
 * `IncreaseReferenceCountRequest` with server. It requires the derived class to
 * implement the:
 *  - `OnRelease(ID)` method to describe what will happens when `ref_count`
 * reaches zero.
 *  - `OnDelete(ID)` method to describe what will happens when `ref_count`
 * reaches zero and the object is marked as to be deleted.
 */
template <typename ID, typename P, typename Der>
class UsageTracker : public LifeCycleTracker<ID, P, UsageTracker<ID, P, Der>> {
 public:
  using base_t = LifeCycleTracker<ID, P, UsageTracker<ID, P, Der>>;
  UsageTracker() {}

  /**
   * @brief Fetch the blob payload from the client-side cache.
   *
   * @param id The object id.
   * @param payload The payload of the object.
   * @returns Status::OK() if the reference count is increased successfully.
   */
  Status FetchOnLocal(ID const& id, P& payload);

  /**
   * @brief Mark the blob in the client-side cache as sealed, to keep
   * consistent with server on blob visibility.
   *
   * @param id The object id.
   */
  Status SealUsage(ID const& id);

  /**
   * @brief Increase the Reference Count, add it to cache if not exists.
   *
   * @param id The object id.
   */
  Status AddUsage(ID const& id, P const& payload);

  /**
   * @brief Decrease the Reference Count.
   *
   * @param id The object id.
   */
  Status RemoveUsage(ID const& id);

  /**
   * @brief Try to delete the blob from the client-side cache.
   *
   * @param id The object id.
   */
  Status DeleteUsage(ID const& id);

  void ClearCache();

  /**
   * @brief change the reference count of the object on the client-side cache.
   */
  Status FetchAndModify(ID const& id, int64_t& ref_cnt, int64_t change);

  Status OnRelease(ID const& id);

  Status OnDelete(ID const& id);

 private:
  inline Der& self() { return static_cast<Der&>(*this); }

  // Track the objects' usage.
  std::unordered_map<ID, std::shared_ptr<P>> object_in_use_;

  friend class LifeCycleTracker<ID, P, Der>;
};

}  // namespace detail

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
                 StoreType const& bulk_store_type = StoreType::kDefault,
                 std::string const& username = "",
                 std::string const& password = "");

  /**
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   * @param bulk_store_type The name of the bulk store.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket,
              StoreType const& bulk_store_type = StoreType::kDefault,
              std::string const& username = "",
              std::string const& password = "");

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
class Client final : public BasicIPCClient,
                     protected detail::UsageTracker<ObjectID, Payload, Client> {
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
   * @brief Connect to vineyard using the UNIX domain socket file specified by
   *        the environment variable `VINEYARD_IPC_SOCKET`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(std::string const& username, std::string const& password);

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
   * @brief Connect to vineyardd using the given UNIX domain socket
   * `ipc_socket`.
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& ipc_socket, std::string const& username,
                 std::string const& password);

  /**
   * @brief Disconnect this client.
   */
  void Disconnect();

  /**
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket);

  /**
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket, std::string const& username,
              std::string const& password);

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
  Status FetchAndGetMetaData(const ObjectID id, ObjectMeta& meta_data,
                             const bool sync_remote = false);

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
  Status GetMetaData(const std::vector<ObjectID>& ids, std::vector<ObjectMeta>&,
                     const bool sync_remote = false);

  /**
   * @brief Create a blob in vineyard server. When creating a blob, vineyard
   * server's bulk allocator will prepare a block of memory of the requested
   * size, then map the memory to client's process to share the allocated
   * memory.
   *
   * @param size The size of requested blob.
   * @param blob The result mutable blob will be set in `blob`.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status CreateBlob(size_t size, std::unique_ptr<BlobWriter>& blob);

  /**
   * @brief Get a blob from vineyard server.
   *
   * @param id the blob to get.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetBlob(ObjectID const id, std::shared_ptr<Blob>& blob);

  /**
   * @brief Get a blob from vineyard server, and optionally bypass the "sealed"
   * check.
   *
   * @param id the blob to get.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetBlob(ObjectID const id, bool unsafe, std::shared_ptr<Blob>& blob);

  /**
   * @brief Get a blob from vineyard server.
   *
   * @param id the blob to get.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetBlobs(std::vector<ObjectID> const ids,
                  std::vector<std::shared_ptr<Blob>>& blobs);

  /**
   * @brief Get a blob from vineyard server, and optionally bypass the "sealed"
   * check.
   *
   * @param id the blob to get.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetBlobs(std::vector<ObjectID> const ids, const bool unsafe,
                  std::vector<std::shared_ptr<Blob>>& blobs);

  /**
   * @brief Claim a shared blob that backed by a file on disk. Users need to
   * provide either a filename to mmap, or an expected size to allocate the
   * file on disk.
   *
   * When mapping existing file as a blob, if the existing file size is
   * smaller than specified "size", the file will be enlarged using
   * ftruncate().
   *
   * Note that when deleting the blob that backed by files, the file won't
   * be automatically deleted by vineyard.
   *
   * @param size expected file size to allocate.
   * @param path use existing file as the mmap buffer.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status CreateDiskBlob(size_t size, const std::string& path,
                        std::unique_ptr<BlobWriter>& blob);

  /**
   * @brief Allocate a chunk of given size in vineyard for a stream. When the
   * request cannot be satisfied immediately, e.g., vineyard doesn't have
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
                            std::unique_ptr<MutableBuffer>& blob);

  // bring the overloadings in parent class to current scope.
  using ClientBase::PullNextStreamChunk;

  /**
   * @brief Pull a chunk from a stream. When there's no more chunk available in
   * the stream, i.e., the stream has been stopped, a status code
   * `kStreamDrained` or `kStreamFinish` will be returned, otherwise the reader
   * will be blocked until writer creates a new chunk in the stream.
   *
   * @param id The id of the stream.
   * @param chunk The immutable chunk generated by the writer of the stream.
   *
   * @return Status that indicates whether the polling has succeeded.
   */
  Status PullNextStreamChunk(ObjectID const id, std::unique_ptr<Buffer>& chunk);

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
   *
   * @return A std::shared_ptr<Object> that can be safely cast to the underlying
   * concrete object type. When the object doesn't exists an std::runtime_error
   * exception will be raised.
   */
  std::shared_ptr<Object> FetchAndGetObject(const ObjectID id);

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
   * @brief Get an object from vineyard. The ObjectFactory will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   * @param object The result object will be set in parameter `object`.
   *
   * @return When errors occur during the request, this method won't throw
   * exceptions, rather, it results a status to represents the error.
   */
  Status FetchAndGetObject(const ObjectID id, std::shared_ptr<Object>& object);

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
   * @param id The object id to get.
   *
   * @return A std::shared_ptr<Object> that can be safely cast to the underlying
   * concrete object type. When the object doesn't exists an std::runtime_error
   * exception will be raised.
   */
  template <typename T>
  std::shared_ptr<T> FetchAndGetObject(const ObjectID id) {
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
      return Status::ObjectTypeError(type_name<T>(),
                                     _object->meta().GetTypeName());
    } else {
      return Status::OK();
    }
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
   *    client.FetchAndGetObject(id, int_array);
   * \endcode
   *
   * @param id The object id to get.
   * @param object The result object will be set in parameter `object`.
   *
   * @return When errors occur during the request, this method won't throw
   * exceptions, rather, it results a status to represents the error.
   */
  template <typename T>
  Status FetchAndGetObject(const ObjectID id, std::shared_ptr<T>& object) {
    std::shared_ptr<Object> _object;
    RETURN_ON_ERROR(FetchAndGetObject(id, _object));
    object = std::dynamic_pointer_cast<T>(_object);
    if (object == nullptr) {
      return Status::ObjectNotExists("object not exists: " +
                                     ObjectIDToString(id));
    } else {
      return Status::OK();
    }
  }

  /**
   * @brief Get multiple objects from vineyard.
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
   * @param target The pointer that been queried.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const void* target) const __attribute__((
      deprecated("Use IsSharedMemory(target, object_id) instead.")));

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * @param target The pointer that been queried.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const uintptr_t target) const __attribute__((
      deprecated("Use IsSharedMemory(target, object_id) instead.")));

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * @param target The pointer that been queried.
   * @param object_id Return the object id of the queried pointer, if found.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const void* target, ObjectID& object_id) const;

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * @param target The pointer that been queried.
   * @param object_id Return the object id of the queried pointer, if found.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const uintptr_t target, ObjectID& object_id) const;

  /**
   * @brief Check if the blob is a cold blob (no client is using it).
   *
   * Return true if the the blob is in-use.
   */
  Status IsInUse(ObjectID const& id, bool& is_in_use);

  /**
   * @brief Check if the blob is a spilled blob (those no client is using and be
   * dumped on disk).
   *
   * Return true if the the blob is spilled.
   */
  Status IsSpilled(ObjectID const& id, bool& is_spilled);

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

  /**
   * @brief Decrease the reference count of the object. It will trigger
   * `OnRelease` behavior when reference count reaches zero. See UsageTracker.
   */
  Status Release(std::vector<ObjectID> const& ids);

  Status Release(ObjectID const& id) override;

  /**
   * @brief Delete metadata in vineyard. When the object is a used by other
   * object, it will be deleted only when the `force` parameter is specified.
   *
   * @param id The ID to delete.
   * @param force Whether to delete the object forcely. Forcely delete an object
   *        means the object and objects which use this object will be delete.
   *        Default is false.
   * @param deep Whether to delete the member of this object. Default is true.
   *        Note that when deleting object which has *direct* blob members, the
   *        processing on those blobs yields a "deep" behavior.
   *
   * @return Status that indicates whether the delete action has succeeded.
   */
  Status DelData(const ObjectID id, const bool force = false,
                 const bool deep = true);
  /**
   * @brief Delete multiple metadatas in vineyard.
   *
   * @param ids The IDs to delete.
   * @param force Whether to delete the object forcely. Forcely delete an object
   *        means the object and objects which use this object will be delete.
   *        Default is false.
   * @param deep Whether to delete the member of this object. Default is true.
   *        Note that when deleting objects which have *direct* blob members,
   *        the processing on those blobs yields a "deep" behavior.
   *
   * @return Status that indicates whether the delete action has succeeded.
   */
  Status DelData(const std::vector<ObjectID>& ids, const bool force = false,
                 const bool deep = true);

  Status CreateGPUBuffer(const size_t size, ObjectID& id, Payload& payload,
                         std::shared_ptr<GPUUnifiedAddress>& gua);
  /**
   * @brief Get a set of blobs from vineyard server. See also `GetBuffer`.
   *
   * @param ids Object ids for the blobs to get.
   * @param buffers: The result result cudaIpcMemhandles related to GPU blobs.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetGPUBuffers(const std::set<ObjectID>& ids, const bool unsafe,
                       std::map<ObjectID, GPUUnifiedAddress>& GUAs);

 protected:
  /**
   * @brief Required by `UsageTracker`. When reference count reaches zero, send
   * the `ReleaseRequest` to server.
   */
  Status OnRelease(ObjectID const& id);

  /**
   * @brief Required by `UsageTracker`. Currently, the deletion does not respect
   * the reference count, it will send the DelData to server and do the deletion
   * forcely.
   */
  Status OnDelete(ObjectID const& id);

  /**
   * @brief Increase reference count after a new object is sealed.
   */
  Status PostSeal(ObjectMeta const& meta_data);

  /**
   * @brief Send request to server to get all underlying blobs of a object.
   */
  Status GetDependency(ObjectID const& id, std::set<ObjectID>& bids);

  Status CreateBuffer(const size_t size, ObjectID& id, Payload& payload,
                      std::shared_ptr<MutableBuffer>& buffer);

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
  Status GetBuffer(const ObjectID id, std::shared_ptr<Buffer>& buffer);

  /**
   * @brief Get a set of blobs from vineyard server. See also `GetBuffer`.
   *
   * @param ids Object ids for the blobs to get.
   * @param buffers: The result immutable blobs will be added to `buffers`.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetBuffers(const std::set<ObjectID>& ids,
                    std::map<ObjectID, std::shared_ptr<Buffer>>& buffers);

  /**
   * @brief Get the size of blobs from vineyard server.
   *
   * @param ids Object ids for the blobs to get.
   * @param sizes: The result sizes of immutable blobs will be added to `sizes`.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
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

  /**
   * @brief mark the blob as sealed to control the visibility of a blob, client
   * can never `Get` an unsealed blob.
   */
  Status Seal(ObjectID const& object_id);

 private:
  Status GetBuffers(const std::set<ObjectID>& ids, const bool unsafe,
                    std::map<ObjectID, std::shared_ptr<Buffer>>& buffers);

  Status GetBufferSizes(const std::set<ObjectID>& ids, const bool unsafe,
                        std::map<ObjectID, size_t>& sizes);

  friend class Blob;
  friend class BlobWriter;
  friend class ObjectBuilder;
  friend class detail::UsageTracker<ObjectID, Payload, Client>;
};

class PlasmaClient final
    : public BasicIPCClient,
      public detail::UsageTracker<PlasmaID, PlasmaPayload, PlasmaClient> {
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
   * @brief Disconnect this client.
   */
  void Disconnect();

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

  Status GetBuffers(std::set<PlasmaID> const& plasma_ids,
                    std::map<PlasmaID, std::shared_ptr<Buffer>>& buffers);

  Status ShallowCopy(PlasmaID const plasma_id, PlasmaID& target_pid,
                     PlasmaClient& source_client);

  Status ShallowCopy(ObjectID const id, std::set<PlasmaID>& target_pids,
                     Client& source_client);

  /**
   * @brief mark the blob as sealed to control the visibility of a blob, client
   * can never `Get` an unsealed blob.
   */
  Status Seal(PlasmaID const& object_id);

  /**
   * @brief Decrease the reference count of a plasma object. It will trigger
   * `OnRelease` behavior when reference count reaches zero. See UsageTracker.
   */
  Status Release(PlasmaID const& id);

  /**
   * @brief Delete a plasma object. It will trigger `OnDelete` behavior when
   * reference count reaches zero. See UsageTracker.
   */
  Status Delete(PlasmaID const& id);

 protected:
  /**
   * @brief Required by `UsageTracker`. When reference count reaches zero, send
   * the `ReleaseRequest` to server.
   */
  Status OnRelease(PlasmaID const& id);

  /**
   * @brief Required by `UsageTracker`. Deletion will be deferred until its
   * reference count reaches zero.
   */
  Status OnDelete(PlasmaID const& id);

  using ClientBase::Release;

 private:
  Status GetPayloads(std::set<PlasmaID> const& plasma_ids, const bool unsafe,
                     std::map<PlasmaID, PlasmaPayload>& plasma_payloads);

  Status GetBuffers(std::set<PlasmaID> const& plasma_ids, const bool unsafe,
                    std::map<PlasmaID, std::shared_ptr<Buffer>>& buffers);

  friend class detail::UsageTracker<PlasmaID, PlasmaPayload, PlasmaClient>;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_CLIENT_H_
