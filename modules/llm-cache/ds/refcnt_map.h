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

#ifndef MODULES_LLM_CACHE_DS_REFCNT_MAP_H_
#define MODULES_LLM_CACHE_DS_REFCNT_MAP_H_

#include <map>
#include <memory>
#include <set>

#include "client/ds/blob.h"

namespace vineyard {

struct MapEntry {
  ObjectID objectID;
  uint64_t refcnt;
};

class RefcntMapObject : public vineyard::Registered<RefcntMapObject> {
 private:
  int size;
  std::shared_ptr<Blob> blob;

 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<RefcntMapObject>{new RefcntMapObject()});
  }

  void Construct(const ObjectMeta& meta) override;

  void Resolve();

  ~RefcntMapObject() = default;

  friend class RefcntMapObjectBuilder;
};

class RefcntMapObjectBuilder : public vineyard::ObjectBuilder {
 private:
  Client& client;
  std::unique_ptr<BlobWriter> mapWriter;
  std::map<ObjectID, uint64_t> refcntMap;

 public:
  explicit RefcntMapObjectBuilder(Client& client);

  RefcntMapObjectBuilder(Client& client,
                         std::shared_ptr<RefcntMapObject>& refcntMap);

  void IncRefcnt(ObjectID objectID);

  void IncSetRefcnt(std::set<ObjectID>& objectIDs);

  void DecRefcnt(ObjectID objectID);

  void DecSetRefcnt(std::set<ObjectID>& objectIDs);

  void PrintRefcntMap();

  bool Equals(std::shared_ptr<RefcntMapObjectBuilder>& refcntMapBuilder);

  std::map<ObjectID, uint64_t> GetRefcntMap();

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& Client) override;

  ~RefcntMapObjectBuilder() = default;
};

}  // namespace vineyard
#endif  // MODULES_LLM_CACHE_DS_REFCNT_MAP_H_
