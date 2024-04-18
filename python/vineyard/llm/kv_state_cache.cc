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

#include <pybind11/cast.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "client/client.h"

#include "llm-cache/ds/config.h"
#include "llm-cache/ds/kv_state_cache_block.h"
#include "llm-cache/ds/kv_state_cache_manager.h"

namespace py = pybind11;

namespace vineyard {

PYBIND11_MODULE(llm_C, m) {
  m.doc() = "vineyard llm kv cache manager module";

  pybind11::enum_<FilesystemType>(m, "FilesystemType")
      .value("LOCAL", FilesystemType::LOCAL)
      .export_values();

  py::class_<LLMKV, std::shared_ptr<LLMKV>>(m, "KVTensor",
                                            py::buffer_protocol())
      .def(py::init([](uintptr_t data, size_t length) {
             return LLMKV{reinterpret_cast<void*>(data), length};
           }),
           py::arg("data"), py::arg("length"))
      .def_property(
          "data",
          [](LLMKV& self) -> uintptr_t {  // getter
            return reinterpret_cast<uintptr_t>(self.data);
          },
          [](LLMKV& self, uintptr_t new_ptr) {  // setter
            self.data = reinterpret_cast<void*>(new_ptr);
          })
      .def_property(
          "length",
          [](LLMKV& self) -> size_t {  // getter
            return self.length;
          },
          [](LLMKV& self, size_t new_length) {  // setter
            self.length = new_length;
          })
      .def_buffer([](LLMKV& self) -> py::buffer_info {
        return py::buffer_info(self.data, sizeof(char),
                               py::format_descriptor<char>::value, 1,
                               {self.length}, {sizeof(char)});
      });

  py::class_<KVStateCacheManager, std::shared_ptr<KVStateCacheManager>>(
      m, "KVStateCacheManager")
      .def(
          "update",
          [](KVStateCacheManager* self, const std::vector<int>& tokenList,
             int& next_token,
             const std::vector<std::pair<vineyard::LLMKV, vineyard::LLMKV>>&
                 kv_state) {
            VINEYARD_CHECK_OK(self->Update(tokenList, next_token, kv_state));
          },
          py::arg("tokens"), py::arg("next_token"), py::arg("kv_state"))
      .def(
          "update",
          [](KVStateCacheManager* self, const std::vector<int>& tokens,
             const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_states)
              -> size_t {
            size_t updated = 0;
            VINEYARD_CHECK_OK(self->Update(tokens, kv_states, updated));
            return updated;
          },
          py::arg("tokens"), py::arg("kv_states"))
      .def(
          "update",
          [](KVStateCacheManager* self, const std::vector<int>& prefix,
             std::vector<int>& tokens,
             const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_states)
              -> size_t {
            size_t updated = 0;
            VINEYARD_CHECK_OK(self->Update(prefix, tokens, kv_states, updated));
            return updated;
          },
          py::arg("prefix"), py::arg("tokens"), py::arg("kv_states"))
      .def(
          "query",
          [](KVStateCacheManager* self, const std::vector<int>& tokens,
             int& next_token, py::list& kv_state) {
            std::vector<std::pair<LLMKV, LLMKV>> kv_state_vec =
                kv_state.cast<std::vector<std::pair<LLMKV, LLMKV>>>();
            VINEYARD_CHECK_OK(self->Query(tokens, next_token, kv_state_vec));
            for (size_t i = 0; i < kv_state_vec.size(); ++i) {
              kv_state[i] = py::cast(kv_state_vec[i]);
            }
          },
          py::arg("tokens"), py::arg("next_token"), py::arg("kv_states"))
      .def(
          "query",
          [](KVStateCacheManager* self, const std::vector<int>& tokens,
             py::list& kv_state_list) -> size_t {
            std::vector<std::vector<std::pair<LLMKV, LLMKV>>> kv_state_vec =
                kv_state_list
                    .cast<std::vector<std::vector<std::pair<LLMKV, LLMKV>>>>();
            size_t matched = 0;
            VINEYARD_CHECK_OK(self->Query(tokens, kv_state_vec, matched));
            for (size_t i = 0; i < kv_state_vec.size() && i < matched; ++i) {
              for (size_t j = 0; j < kv_state_vec[i].size(); ++j) {
                kv_state_list[i].cast<py::list>()[j] =
                    py::cast(kv_state_vec[i][j]);
              }
            }
            return matched;
          },
          py::arg("tokens"), py::arg("kv_states"))
      .def("close", [](KVStateCacheManager* self) { self->Close(); });

  m.def(
       "_generate",
       [](py::object ipc_client, int tensor_bytes, int cache_capacity,
          int layer, int block_size, int sync_interval,
          std::string llm_cache_sync_lock, std::string llm_cache_object_name,
          std::string llm_ref_cnt_object_name)
           -> std::shared_ptr<KVStateCacheManager> {
         std::shared_ptr<KVStateCacheManager> manager;
         VineyardCacheConfig config(tensor_bytes, cache_capacity, layer,
                                    block_size, sync_interval,
                                    llm_cache_sync_lock, llm_cache_object_name,
                                    llm_ref_cnt_object_name);
         Client& client = ipc_client.cast<Client&>();
         vineyard::Status status =
             vineyard::KVStateCacheManager::Make(client, manager, config);
         if (!status.ok()) {
           throw std::runtime_error(status.ToString());
         }
         return manager;
       },
       py::arg("ipc_client"), py::arg("tensor_bytes") = 10,
       py::arg("cache_capacity") = 10, py::arg("layer") = 1,
       py::arg("block_size") = 5, py::arg("sync_interval") = 3,
       py::arg("llm_cache_sync_lock") = "llmCacheSyncLock",
       py::arg("llm_cache_object_name") = "llm_cache_object",
       py::arg("llm_ref_cnt_object_name") = "llm_refcnt_object")
      .def(
          "_generate",
          [](int tensor_bytes, int cache_capacity, int layer, int chunk_size,
             int split_number, std::string root, FilesystemType filesystemType,
             int client_gc_interval, int ttl, bool enable_global_gc, int global_gc_interval,
             int global_ttl)
              -> std::shared_ptr<KVStateCacheManager> {
            std::shared_ptr<KVStateCacheManager> manager;
            FileCacheConfig config(tensor_bytes, cache_capacity, layer,
                                   chunk_size, split_number, root,
                                   filesystemType, client_gc_interval, ttl,
                                   enable_global_gc, global_gc_interval, global_ttl);
            VINEYARD_CHECK_OK(
                vineyard::KVStateCacheManager::Make(manager, config));
            return manager;
          },
          py::arg("tensor_bytes") = 10, py::arg("cache_capacity") = 10,
          py::arg("layer") = 1, py::arg("chunk_size") = 5,
          py::arg("split_number") = 3, py::arg("root") = "root",
          py::arg("filesystem_type") = FilesystemType::LOCAL,
          py::arg("client_gc_interval") = 30 * 60, py::arg("ttl") = 30 * 60,
          py::arg("enable_global_gc") = false, py::arg("global_gc_interval") = 30 * 60,
          py::arg("global_ttl") = 30 * 60);
}

}  // namespace vineyard
