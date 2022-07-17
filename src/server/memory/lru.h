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

#ifndef SRC_SERVER_MEMORY_LRU_H_
#define SRC_SERVER_MEMORY_LRU_H_

#include "oneapi/tbb/concurrent_hash_map.h"
#include "common/util/status.h"
#include "common/util/logging.h"

#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <mutex>
#include <list>

namespace vineyard{
  namespace detail{
    template<typename ID, typename P>
    class LRU {
    public:
      using value_t = std::pair<ID, std::shared_ptr<P>>;
      using lru_map_t =
          std::unordered_map<ID, typename std::list<value_t>::iterator>;
      using lru_list_t = std::list<value_t>;
      LRU() = default;
      ~LRU() = default;
      void Ref(ID id, std::shared_ptr<P> payload) {
        std::unique_lock<decltype(mu_)> locked;
        auto it = map_.find(id);
        if (it == map_.end()) {
          list_.emplace_front(id, payload);
          map_.emplace(id, list_.begin());
        } else {
          list_.erase(it->second);
          list_.emplace_front(id, payload);
          it->second = list_.begin();
        }
      }

      bool CheckExist(ID id) const {
        std::shared_lock<decltype(mu_)> shared_locked;
        auto it = map_.find(id);
        if (it == map_.end()) {
          return false;
        }
        return true;
      }

      /**
      * @brief Here we have two actions: 1. delete from lru_list
      *        2. delete from spilled_obj_
      * @param id
      * @return Status
      */
      Status Unref(const ID& id, std::shared_ptr<Der> store_ptr) {
        std::unique_lock<decltype(mu_)> locked;
        auto it = map_.find(id);
        if (it == map_.end()) {
          auto it = spilled_obj_.find(id);
          if (it == spilled_obj_.end()) {
            return Status::OK();
          }
          RETURN_ON_ERROR(it->second->ReloadFromSpill(store_ptr));
          spilled_obj_.erase(it);
          return Status::OK();
        }
        list_.erase(it->second);
        map_.erase(it);
        return Status::OK();
      }

      /**
      * @brief spill cold-obj till their sizes sum up to sz
      *
      * @param sz: specify the spilled size
      * @return: if objects are spilled
      */
      Status Spill(size_t sz, std::shared_ptr<BulkStore> bulk_store_ptr) {
        std::unique_lock<decltype(mu_)> locked;
        size_t spilled_sz = 0;
        auto st = Status::OK();
        LOG(INFO) << "vineryardd is trying to spilling objects for more space...";
        for (auto it = list_.begin(); it != list_.end();) {
          LOG(INFO) << "\tspilling ObjID: " << it->first;
          st = it->second->Spill();
          if (!st.ok()) { 
            LOG(ERROR) << st.ToString();
            break;
          }
          spilled_sz += it->second->data_size;
          LOG(ERROR) << "Gonna spill ObjectID: " << it->first;
          spilled_obj_.emplace(it->first, it->second);
          LOG(INFO) << "spill_obj now contains:"; 
          for(auto iter = spilled_obj_.begin(); iter != spilled_obj_.end(); iter++){
            LOG(INFO) <<"\t" << iter->first; 
          }
          map_.erase(it->first);
          // erase here is safe for list's iterator
          it = list_.erase(it);
          if (sz <= spilled_sz) {
            break;
          }
        }
        if(st.ok() && spilled_sz == 0){
          return Status::NotEnoughMemory("Nothing spilled");
        }
        return st; 
      }

      bool CheckSpilled(const ID& id){
        std::shared_lock<decltype(mu_)> lock;
        LOG(INFO) << "Searching for " << id;
        return spilled_obj_.find(id) != spilled_obj_.end();
      }

    private:
  #if __APPLE__
      mutable boost::shared_mutex mu_;
  #else
      mutable std::shared_timed_mutex mu_;
  #endif
      // protected by mu_
      lru_map_t map_;
      lru_list_t list_;
      std::unordered_map<ID, std::shared_ptr<P>> spilled_obj_;
    };

  }
}

#endif