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

#ifndef SRC_CLIENT_DS_I_OBJECT_H_
#define SRC_CLIENT_DS_I_OBJECT_H_

#include <memory>

#include "client/ds/object_factory.h"
#include "client/ds/object_meta.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class ClientBase;
class Client;
class RPCClient;

class ObjectBase;
class Object;
class ObjectBuilder;

/**
 * @brief ObjectBase is the most base class for vineyard's Object and
 * ObjectBuilder.
 *
 * An ObjectBase instance is a build-able value that it's `Build` method put
 * blobs * into vineyard server, and it's `_Seal` method is responsible for
 * creating metadata in vineyard server for the object.
 */
class ObjectBase {
 public:
  /**
   * @brief Building the object means construct all blobs of this object to
   * vineyard server.
   *
   * @param client The vineyard client that been used to create blobs in the
   * connected vineyard server.
   */
  virtual Status Build(Client& client) = 0;

  /**
   * @brief Sealing the object means construct the metadata for the object and
   * create metadata in vineyard server.
   *
   * @param client The vineyard client that been used to create metadata in the
   * connected vineyard server.
   */
  virtual std::shared_ptr<Object> _Seal(Client& client) = 0;
};

/**
 * @brief Object is the core concept in vineyard. Object can be a scalar, a
 * tuple, a vector, a tensor, or even a distributed graph in vineyard. Objects
 * are stored in vineyard, and can be shared to process that connects to the
 * same vineyard cluster.
 *
 * Every object in vineyard has a unique identifier `ObjectID` that can be
 * passed back and forth in the computation pipeline. An object is composed by a
 * metadata, and a set of blobs.
 *
 * Object in vineyard is by-design to be hierarchical, and can have other Object
 * as members. For example, a tensor object may has a vector as its payload, and
 * a distributed dataframe has many dataframe objects as its chunks, and a
 * dataframe is composed by an array of tensors as columns.
 */
class Object : public ObjectBase, public std::enable_shared_from_this<Object> {
  // NB: the std::enable_shared_from_this inheritance must be public
 public:
  virtual ~Object();

  /**
   * @brief The object id of this object.
   */
  ObjectID const id() const;

  /**
   * @brief The metadata of this object.
   */
  const ObjectMeta& meta() const;

  /**
   * @brief The nbytes of this object, can be treated as the memory usage of
   * this object.
   */
  size_t const nbytes() const;

  /**
   * @brief Construct an object from metadata. The metadata `meta` should come
   * from client's GetMetaData method.
   *
   * The implementation of `Construct` method is usually boilerplate. Vineyard
   * provides a code generator to help developers code their own data structures
   * and can be shared via vineyard.
   *
   * @param meta The metadata that be used to construct the object.
   */
  virtual void Construct(const ObjectMeta& meta);

  /**
   * @brief `PostConstruct` is called at the end of `Construct` to perform
   * user-specific constructions that the code generator cannot handle.
   *
   * @param meta The metadata that be used to construct the object.
   */
  virtual void PostConstruct(const ObjectMeta& meta) {}

  /**
   * @brief Object is also a kind of ObjectBase, and can be used as a member to
   * construct new objects. The Object type also has a `Build` method but it
   * does nothing, though.
   */
  Status Build(Client& client) final { return Status::OK(); }

  /**
   * @brief Object is also a kind of ObjectBase, and can be used as a member to
   * construct new objects. The Object type also has a `_Seal` method but it
   * does nothing, though.
   */
  std::shared_ptr<Object> _Seal(Client& client) final;

  /**
   * @brief Persist the object to make it visible for clients that connected to
   * other vineyardd instances in the cluster.
   *
   * @param client The client that to be used to perform the `Persist` request.
   */
  Status Persist(ClientBase& client) const;

  /**
   * @brief The verb "local" means it is a local object to the client. A local
   * object's blob can be accessed by the client in a zero-copy fashion.
   * Otherwise only the metadata is accessible.
   *
   * @return True iff the object is a local object.
   */
  bool const IsLocal() const;

  /**
   * @brief The verb "persist" means it is visible for client that connected to
   * other vineyardd instances in the cluster. After `Persist` the object
   * becomes persistent.
   *
   * @return True iff the object is a persistent object.
   */
  bool const IsPersist() const;

  /**
   * @brief The verb "global" means it is a global object and only refers some
   * local objects.
   *
   * @return True iff the object is a global object.
   */
  bool const IsGlobal() const;

 protected:
  Object() {}

  ObjectID id_;
  mutable ObjectMeta meta_;

  friend class ClientBase;
  friend class Client;
  friend class PlasmaClient;
  friend class RPCClient;
  friend class ObjectMeta;
};

/**
 * Global object is an tag class to mark a type as a vineyard's GlobalObject.
 *
 * User-defined global object types should inherit this tag class.
 */
struct GlobalObject {};

class ObjectBuilder : public ObjectBase {
 public:
  virtual ~ObjectBuilder() {}

  Status Build(Client& client) override = 0;

  virtual std::shared_ptr<Object> Seal(Client& client);

  virtual Status Seal(Client& client, std::shared_ptr<Object>& object);

  //  protected: FIXME
  std::shared_ptr<Object> _Seal(Client& client) override;

  //  protected: FIXME
  virtual Status _Seal(Client& client, std::shared_ptr<Object>& object);

  bool sealed() const { return sealed_; }

 protected:
  void set_sealed(bool const sealed = true) { this->sealed_ = sealed; }

 private:
  bool sealed_ = false;
};

/**
 * @brief Register a type as vineyard Object type by inheriting Registered.
 */
template <typename T>
class __attribute__((visibility("default"))) Registered : public Object {
 protected:
  __attribute__((visibility("default"))) Registered() {
    FORCE_INSTANTIATE(registered);
  }

 private:
  __attribute__((visibility("default"))) static const bool registered;
};

template <typename T>
const bool Registered<T>::registered = ObjectFactory::Register<T>();

/**
 * @brief Register a type as vineyard Object type, without inherits `Object`, by
 * inheriting BareRegistered.
 */
template <typename T>
class __attribute__((visibility("default"))) BareRegistered {
 protected:
  __attribute__((visibility("default"))) BareRegistered() {
    FORCE_INSTANTIATE(registered);
  }

 private:
  __attribute__((visibility("default"))) static const bool registered;
};

template <typename T>
const bool BareRegistered<T>::registered = ObjectFactory::Register<T>();

/**
 * @brief Throws an exception if the builder has already been sealed.
 */
#ifndef ENSURE_NOT_SEALED
#define ENSURE_NOT_SEALED(builder)                                \
  do {                                                            \
    if (builder->sealed()) {                                      \
      std::clog << "[error] The builder has already been sealed"; \
      VINEYARD_CHECK_OK(vineyard::Status::ObjectSealed(           \
          "The builder has already been sealed"));                \
    }                                                             \
  } while (0)
#endif  // ENSURE_NOT_SEALED

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_I_OBJECT_H_
