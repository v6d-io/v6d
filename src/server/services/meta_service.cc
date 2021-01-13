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

#include "server/services/meta_service.h"

#include <memory>

#include "glog/logging.h"

#include "server/services/etcd_meta_service.h"
#include "server/util/meta_tree.h"

namespace vineyard {

std::shared_ptr<IMetaService> IMetaService::Get(vs_ptr_t ptr) {
  return std::shared_ptr<IMetaService>(new EtcdMetaService(ptr));
}

void IMetaService::incRef(std::string const& key, std::string const& value) {
  std::vector<std::string> vs;
  boost::algorithm::split(vs, key, [](const char c) { return c == '/'; });
  if (vs[0].empty()) {
    vs.erase(vs.begin());
  }
  if (vs.size() < 2 || vs[0] != "data") {
    // The key is not an object id: data.id
    return;
  }
  ObjectID key_obj, value_obj;
  if (meta_tree::DecodeObjectID(value, value_obj).ok()) {
    key_obj = VYObjectIDFromString(vs[1]);
    {
      // validate the dependency graph
      decltype(subobjects_.begin()) iter;
      auto range = subobjects_.equal_range(key_obj);
      for (iter = range.first; iter != range.second; ++iter) {
        if (iter->second == value_obj) {
          break;
        }
      }
      if (iter == range.second) {
        subobjects_.emplace(key_obj, value_obj);
      }
    }
    {
      // validate the dependency graph
      decltype(supobjects_.begin()) iter;
      auto range = supobjects_.equal_range(value_obj);
      for (iter = range.first; iter != range.second; ++iter) {
        if (iter->second == key_obj) {
          break;
        }
      }
      if (iter == range.second) {
        supobjects_.emplace(value_obj, key_obj);
      }
    }
  }
}

bool IMetaService::deleteable(ObjectID const object_id) {
  return object_id != InvalidObjectID() &&
         supobjects_.find(object_id) == supobjects_.end();
}

void IMetaService::traverseToDelete(std::set<ObjectID>& initial_delete_set,
                                    std::set<ObjectID>& delete_set,
                                    const ObjectID object_id, const bool force,
                                    const bool deep) {
  // emulate a topological sort to ensure the correctness when deleting multiple
  // objects at the same time.
  if (delete_set.find(object_id) != delete_set.end()) {
    // already been processed
    return;
  }
  auto sup_target_range = supobjects_.equal_range(object_id);
  std::set<ObjectID> sup_traget_to_preprocess;
  for (auto it = sup_target_range.first; it != sup_target_range.second; ++it) {
    if (initial_delete_set.find(it->second) != initial_delete_set.end()) {
      sup_traget_to_preprocess.emplace(it->second);
    }
  }
  for (ObjectID const& sup_target : sup_traget_to_preprocess) {
    traverseToDelete(initial_delete_set, delete_set, sup_target, force, deep);
  }
  if (force || deleteable(object_id)) {
    delete_set.emplace(object_id);
    {
      std::set<ObjectID> to_delete;
      auto range = subobjects_.equal_range(object_id);
      for (auto it = range.first; it != range.second; ++it) {
        // remove dependency edge
        auto suprange = supobjects_.equal_range(it->second);
        decltype(suprange.first) p;
        for (p = suprange.first; p != suprange.second; /* no self-inc */) {
          if (p->second == object_id) {
            supobjects_.erase(p++);
          } else {
            ++p;
          }
        }
        if (deep) {
          to_delete.emplace(it->second);
        }
      }
      if (deep) {
        for (auto const& target : to_delete) {
          traverseToDelete(initial_delete_set, delete_set, target, false, true);
        }
      }
    }
    if (force) {
      std::set<ObjectID> to_delete;
      auto range = supobjects_.equal_range(object_id);
      for (auto it = range.first; it != range.second; ++it) {
        // remove dependency edge
        auto subrange = subobjects_.equal_range(it->second);
        decltype(subrange.first) p;
        for (p = subrange.first; p != subrange.second; /* no self-inc */) {
          if (p->second == object_id) {
            subobjects_.erase(p++);
          } else {
            ++p;
          }
        }
        if (force) {
          to_delete.emplace(it->second);
        }
      }
      if (force) {
        for (auto const& target : to_delete) {
          traverseToDelete(initial_delete_set, delete_set, target, true, false);
        }
      }
    }
    subobjects_.erase(object_id);
    supobjects_.erase(object_id);
  }
  initial_delete_set.erase(object_id);
}

void IMetaService::printDepsGraph() {
  std::stringstream ss;
  ss << "object top -> down dependencies: " << std::endl;
  for (auto const& kv : subobjects_) {
    ss << VYObjectIDToString(kv.first) << " -> "
       << VYObjectIDToString(kv.second) << std::endl;
  }
  ss << "object down <- top dependencies: " << std::endl;
  for (auto const& kv : supobjects_) {
    ss << VYObjectIDToString(kv.first) << " <- "
       << VYObjectIDToString(kv.second) << std::endl;
  }
  VLOG(10) << "Depenencies graph:\n" << ss.str();
}

}  // namespace vineyard
