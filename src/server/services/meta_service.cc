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

#include "server/services/meta_service.h"

#include <algorithm>
#include <memory>

#include "glog/logging.h"

#include "server/services/etcd_meta_service.h"
#include "server/services/local_meta_service.h"
#include "server/util/meta_tree.h"

namespace vineyard {

std::shared_ptr<IMetaService> IMetaService::Get(vs_ptr_t server_ptr) {
  std::string meta = server_ptr->GetSpec()["metastore_spec"]["meta"]
                         .get_ref<const std::string&>();
  VINEYARD_ASSERT(meta == "etcd" || meta == "local",
                  "Invalid metastore: " + meta);
  if (meta == "etcd") {
    return std::shared_ptr<IMetaService>(new EtcdMetaService(server_ptr));
  }
  if (meta == "local") {
    return std::shared_ptr<IMetaService>(new LocalMetaService(server_ptr));
  }
  return nullptr;
}

/** Note [Deleting objects and blobs]
 *
 * Blob is special: suppose A -> B and A -> C, where A is an object, B is an
 * object as well, but C is a blob, when delete(A, deep=False), B will won't be
 * touched but C will be checked (and possible deleted) as well.
 *
 * That is because, for remote object, when it being delete (with deep), the
 * deletion of the blobs cannot be reflected in watched etcd events.
 */

/**
 * Note [Deleting global object and members]
 *
 * Because of migration deleting member of global objects is tricky, as, assume
 * we have
 *
 *    A -> B      -- h1
 *      -> C      -- h2
 *
 * then migration happens,
 *
 *    A -> B      -- h1
 *      -> C      -- h2
 *      -> C'     -- h1
 *
 * and if we delete C with (deep=true, force=false), A, B and C' should still be
 * kept as they still construct a complete object.
 *
 * To archive this, we
 *
 * - in `incRef(A, B)`: if `B` is a signature, we just record the object id
 *   instead.
 *
 * - in `findDeleteSet` (in `deleteable`), an object is deleteable, when
 *
 *      * it is not used, or
 *      * it has an *equivalent* object in the metatree, whether it is a local
 *        or remote object. Here *equivalent* means they has the same signature.
 *
 *   Note that in the second case, the signature of *equivalent* object must has
 *   already been pushed into "/signatures/", as we have another assumption that
 *   only member of global objects can be migrated, see also `MigrateObject` in
 *   vineyard_server.cc.
 */

void IMetaService::IncRef(std::string const& instance_name,
                          std::string const& key, std::string const& value,
                          const bool from_remote) {
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
  if (meta_tree::DecodeObjectID(meta_, instance_name, value, value_obj).ok()) {
    key_obj = ObjectIDFromString(vs[1]);
    if (from_remote && IsBlob(value_obj)) {
      // don't put remote blob refs into deps graph, since two blobs may share
      // the same object id.
      return;
    }
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

void IMetaService::CloneRef(ObjectID const target, ObjectID const mirror) {
  // avoid repeatedly clone
  VLOG(10) << "clone ref: " << ObjectIDToString(target) << " -> "
           << ObjectIDToString(mirror);
  if (supobjects_.find(mirror) != supobjects_.end()) {
    return;
  }
  auto range = supobjects_.equal_range(target);
  std::vector<ObjectID> suprefs;
  // n.b.: avoid traverse & modify at the same time (in the same loop).
  for (auto iter = range.first; iter != range.second; ++iter) {
    suprefs.emplace_back(iter->second);
  }
  for (auto const supref : suprefs) {
    supobjects_.emplace(mirror, supref);
    subobjects_.emplace(supref, mirror);
  }
}

bool IMetaService::deleteable(ObjectID const object_id) {
  if (object_id == InvalidObjectID()) {
    return true;
  }
  if (supobjects_.find(object_id) == supobjects_.end()) {
    return true;
  }
  ObjectID equivalent = InvalidObjectID();
  return meta_tree::HasEquivalent(meta_, object_id, equivalent);
}

void IMetaService::traverseToDelete(std::set<ObjectID>& initial_delete_set,
                                    std::set<ObjectID>& delete_set,
                                    int32_t depth,
                                    std::map<ObjectID, int32_t>& depthes,
                                    const ObjectID object_id, const bool force,
                                    const bool deep) {
  // emulate a topological sort to ensure the correctness when deleting multiple
  // objects at the same time.
  if (delete_set.find(object_id) != delete_set.end()) {
    // already been processed
    if (depthes[depth] < depth) {
      depthes[depth] = depth;
    }
    return;
  }
  // process the "initial_delete_set" in topo-sort order.
  auto sup_target_range = supobjects_.equal_range(object_id);
  std::set<ObjectID> sup_traget_to_preprocess;
  for (auto it = sup_target_range.first; it != sup_target_range.second; ++it) {
    if (initial_delete_set.find(it->second) != initial_delete_set.end()) {
      sup_traget_to_preprocess.emplace(it->second);
    }
  }
  for (ObjectID const& sup_target : sup_traget_to_preprocess) {
    traverseToDelete(initial_delete_set, delete_set, depth + 1, depthes,
                     sup_target, force, deep);
  }
  if (force || deleteable(object_id)) {
    delete_set.emplace(object_id);
    depthes[object_id] = depth;
    {
      // delete downwards
      std::set<ObjectID> to_delete;
      {
        // delete sup-edges of subobjects
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
          if (deep || IsBlob(it->second)) {
            // blob is special: see Note [Deleting objects and blobs].
            to_delete.emplace(it->second);
          }
        }
      }

      {
        // delete sub-edges of supobjects
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
        }
      }

      for (auto const& target : to_delete) {
        traverseToDelete(initial_delete_set, delete_set, depth - 1, depthes,
                         target, false, true);
      }
    }
    if (force) {
      // delete upwards
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
          traverseToDelete(initial_delete_set, delete_set, depth + 1, depthes,
                           target, true, false);
        }
      }
    }
    subobjects_.erase(object_id);
    supobjects_.erase(object_id);
  }
  if (initial_delete_set.find(object_id) != initial_delete_set.end()) {
    initial_delete_set.erase(object_id);
  }
}

void IMetaService::findDeleteSet(std::vector<ObjectID> const& object_ids,
                                 std::vector<ObjectID>& processed_delete_set,
                                 bool force, bool deep) {
  // implements dependent-based (usage-based) lifecycle: find the delete set.
  std::set<ObjectID> initial_delete_set{object_ids.begin(), object_ids.end()};
  std::set<ObjectID> delete_set;
  std::map<ObjectID, int32_t> depthes;
  for (auto const object_id : object_ids) {
    traverseToDelete(initial_delete_set, delete_set, 0, depthes, object_id,
                     force, deep);
  }
  postProcessForDelete(delete_set, depthes, processed_delete_set);
}

/**
 * N.B.: all object ids are guaranteed to exist in `depthes`.
 */
void IMetaService::postProcessForDelete(
    const std::set<ObjectID>& delete_set,
    const std::map<ObjectID, int32_t>& depthes,
    std::vector<ObjectID>& delete_objects) {
  delete_objects.assign(delete_set.begin(), delete_set.end());
  std::stable_sort(delete_objects.begin(), delete_objects.end(),
                   [&depthes](const ObjectID& x, const ObjectID& y) {
                     return depthes.at(x) > depthes.at(y);
                   });
}

void IMetaService::printDepsGraph() {
  if (!VLOG_IS_ON(100)) {
    return;
  }
  std::stringstream ss;
  ss << "object top -> down dependencies: " << std::endl;
  for (auto const& kv : subobjects_) {
    ss << ObjectIDToString(kv.first) << " -> " << ObjectIDToString(kv.second)
       << std::endl;
  }
  ss << "object down <- top dependencies: " << std::endl;
  for (auto const& kv : supobjects_) {
    ss << ObjectIDToString(kv.first) << " <- " << ObjectIDToString(kv.second)
       << std::endl;
  }
  VLOG(100) << "Depenencies graph on " << server_ptr_->instance_name() << ": \n"
            << ss.str();
}

void IMetaService::putVal(const kv_t& kv, bool const from_remote) {
  // don't crash the server for any reason (any potential garbage value)
  auto upsert_to_meta = [&]() -> Status {
    json value = json::parse(kv.value);
    if (value.is_string()) {
      IncRef(server_ptr_->instance_name(), kv.key,
             value.get_ref<std::string const&>(), from_remote);
    } else if (value.is_object() && !value.empty()) {
      for (auto const& item : value.items()) {
        if (item.value().is_string()) {
          IncRef(server_ptr_->instance_name(), kv.key,
                 item.value().get_ref<std::string const&>(), from_remote);
        }
      }
    }
    meta_[json::json_pointer(kv.key)] = value;
    return Status::OK();
  };

  auto upsert_sig_to_meta = [&]() -> Status {
    json value = json::parse(kv.value);
    if (value.is_string()) {
      ObjectID object_id =
          ObjectIDFromString(value.get_ref<std::string const&>());
      ObjectID equivalent = InvalidObjectID();
      if (meta_tree::HasEquivalent(meta_, object_id, equivalent)) {
        CloneRef(equivalent, object_id);
      }
    } else {
      LOG(ERROR) << "Invalid signature record: " << kv.key << " -> "
                 << kv.value;
    }
    return Status::OK();
  };

  // update signatures
  if (boost::algorithm::starts_with(kv.key, "/signatures/")) {
    if (!from_remote || !meta_.contains(json::json_pointer(kv.key))) {
      VINEYARD_LOG_ERROR(CATCH_JSON_ERROR(upsert_to_meta()));
    }
    VINEYARD_LOG_ERROR(CATCH_JSON_ERROR(upsert_sig_to_meta()));
    return;
  }

  // update names
  if (boost::algorithm::starts_with(kv.key, "/names/")) {
    if (!from_remote && meta_.contains(json::json_pointer(kv.key))) {
      LOG(WARNING) << "Warning: name got overwritten: " << kv.key;
    }
    VINEYARD_LOG_ERROR(CATCH_JSON_ERROR(upsert_to_meta()));
    return;
  }

  // update ordinary data
  VINEYARD_LOG_ERROR(CATCH_JSON_ERROR(upsert_to_meta()));
}

void IMetaService::delVal(std::string const& key) {
  auto path = json::json_pointer(key);
  if (meta_.contains(path)) {
    auto ppath = path.parent_pointer();
    meta_[ppath].erase(path.back());
    if (meta_[ppath].empty()) {
      meta_[ppath.parent_pointer()].erase(ppath.back());
    }
  }
}

void IMetaService::delVal(const kv_t& kv) { delVal(kv.key); }

void IMetaService::delVal(ObjectID const& target, std::set<ObjectID>& blobs) {
  if (target == InvalidObjectID()) {
    return;
  }
  auto targetkey = json::json_pointer("/data/" + ObjectIDToString(target));
  if (deleteable(target)) {
    // if deletable blob: delete blob
    if (IsBlob(target)) {
      blobs.emplace(target);
    }
    delVal(targetkey);
  } else if (target != InvalidObjectID()) {
    // mark as transient
    meta_[targetkey]["transient"] = true;
  }
}

}  // namespace vineyard
