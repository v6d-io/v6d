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

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "client/client.h"
#include "common/util/logging.h"
#include "llm-cache/ds/refcnt_map.h"

namespace vineyard {

void RefcntMapObject::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);

  std::string typeName = type_name<RefcntMapObject>();

  VINEYARD_ASSERT(meta.GetTypeName() == typeName,
                  "Expect typename '" + typeName + "', but got '" +
                      meta.GetTypeName() + "'");

  size = meta.GetKeyValue<int>("size");
  this->blob = std::dynamic_pointer_cast<Blob>(meta.GetMember("buffer_"));
}

RefcntMapObjectBuilder::RefcntMapObjectBuilder(Client& client)
    : client(client) {}

RefcntMapObjectBuilder::RefcntMapObjectBuilder(
    Client& client, std::shared_ptr<RefcntMapObject>& refcntMapObject)
    : client(client) {
  const MapEntry* mapEntries =
      reinterpret_cast<const MapEntry*>(refcntMapObject->blob->data());
  for (int i = 0; i < refcntMapObject->size; i++) {
    refcntMap[mapEntries[i].objectID] = mapEntries[i].refcnt;
  }
}

void RefcntMapObjectBuilder::IncRefcnt(ObjectID objectID) {
  VLOG(100) << "inc refcnt of :" << objectID;
  if (refcntMap.find(objectID) == refcntMap.end()) {
    refcntMap[objectID] = 1;
  } else {
    refcntMap[objectID]++;
  }
}

void RefcntMapObjectBuilder::IncSetRefcnt(std::set<ObjectID>& objectIDs) {
  for (auto objectID : objectIDs) {
    VLOG(100) << "inc refcnt of :" << objectID;
    if (refcntMap.find(objectID) == refcntMap.end()) {
      refcntMap[objectID] = 1;
    } else {
      refcntMap[objectID]++;
    }
  }
}

void RefcntMapObjectBuilder::DecRefcnt(ObjectID objectID) {
  VLOG(100) << "dec refcnt of :" << objectID;
  if (refcntMap.find(objectID) != refcntMap.end()) {
    refcntMap[objectID]--;
    if (refcntMap[objectID] == 0) {
      // TODO: delete object
      refcntMap.erase(objectID);
      Status status = client.DelData(objectID);
      if (!status.ok()) {
        LOG(ERROR) << "Delete object failed. It may cause memory leak.";
      }
    }
  }
}

void RefcntMapObjectBuilder::DecSetRefcnt(std::set<ObjectID>& objectIDs) {
  std::vector<ObjectID> objectIDToDelete;
  for (auto objectID : objectIDs) {
    VLOG(100) << "dec refcnt of :" << objectID;
    if (refcntMap.find(objectID) != refcntMap.end()) {
      refcntMap[objectID]--;
      if (refcntMap[objectID] == 0) {
        // TODO: delete object
        refcntMap.erase(objectID);
        objectIDToDelete.push_back(objectID);
      }
    }
  }
  if (objectIDToDelete.size() > 0) {
    Status status = client.DelData(objectIDToDelete);
    if (!status.ok()) {
      LOG(ERROR) << "Delete object failed. It may cause memory leak.";
    }
  }
}

bool RefcntMapObjectBuilder::Equals(
    std::shared_ptr<RefcntMapObjectBuilder>& refcntMapBuilder) {
  std::map<ObjectID, uint64_t> refcntMapCompare = refcntMapBuilder->refcntMap;
  if (refcntMap.size() != refcntMapCompare.size()) {
    return false;
  }

  for (auto& entry : refcntMap) {
    if (refcntMapCompare.find(entry.first) == refcntMapCompare.end()) {
      return false;
    }
    if (refcntMapCompare[entry.first] != entry.second) {
      return false;
    }
  }

  return true;
}

void RefcntMapObjectBuilder::PrintRefcntMap() {
  VLOG(100) << "refcntMap size:" << refcntMap.size();
  for (auto& entry : refcntMap) {
    VLOG(100) << "objectID : " << entry.first << " refcnt : " << entry.second;
  }
}

std::map<ObjectID, uint64_t> RefcntMapObjectBuilder::GetRefcntMap() {
  return refcntMap;
}

Status RefcntMapObjectBuilder::Build(Client& client) {
  size_t size = refcntMap.size();
  RETURN_ON_ERROR(client.CreateBlob(size * sizeof(MapEntry), mapWriter));
  MapEntry* mapEntries = reinterpret_cast<MapEntry*>(mapWriter->data());

  int i = 0;
  for (auto& entry : refcntMap) {
    mapEntries[i].objectID = entry.first;
    mapEntries[i].refcnt = entry.second;
    i++;
  }
  return Status::OK();
}

std::shared_ptr<Object> RefcntMapObjectBuilder::_Seal(Client& client) {
  VINEYARD_CHECK_OK(Build(client));

  std::shared_ptr<RefcntMapObject> refcntMapObject =
      std::make_shared<RefcntMapObject>();

  std::shared_ptr<ObjectBase> buffer_ =
      std::shared_ptr<BlobWriter>(std::move(mapWriter));
  refcntMapObject->meta_.AddMember("buffer_", buffer_->_Seal(client));
  refcntMapObject->meta_.AddKeyValue("size", refcntMap.size());
  refcntMapObject->meta_.SetTypeName(type_name<RefcntMapObject>());

  VINEYARD_CHECK_OK(
      client.CreateMetaData(refcntMapObject->meta_, refcntMapObject->id_));
  this->set_sealed(true);
  return refcntMapObject;
}

}  // namespace vineyard
