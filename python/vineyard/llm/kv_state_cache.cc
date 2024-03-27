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

  py::class_<LLMKV>(m, "KVTensor", py::buffer_protocol())
      .def(py::init([](uintptr_t data, size_t length) {
             return LLMKV{reinterpret_cast<void*>(data), length};
           }),
           py::arg("data"), py::arg("length"))
      .def_readwrite("length", &LLMKV::length)
      .def_buffer([](LLMKV& self) -> py::buffer_info {
        return py::buffer_info(self.data, sizeof(char),
                               py::format_descriptor<char>::value, 1,
                               {self.length}, {sizeof(char)});
      });

  py::class_<KVStateCacheManager, std::shared_ptr<KVStateCacheManager>>(
      m, "KVStateCacheManager")
      .def(
          "update",
          [](KVStateCacheManager* self, std::vector<int>& tokenList,
             int& next_token,
             const std::vector<std::pair<vineyard::LLMKV, vineyard::LLMKV>>&
                 kv_state) {
            VINEYARD_CHECK_OK(self->Update(tokenList, next_token, kv_state));
          },
          py::arg("tokens"), py::arg("next_token"), py::arg("kv_state"))
      .def(
          "update",
          [](KVStateCacheManager* self, std::vector<int>& tokens,
             const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>&
                 kv_states) -> size_t {
            size_t updated = 0;
            VINEYARD_CHECK_OK(self->Update(tokens, kv_states, updated));
            return updated;
          },
          py::arg("tokens"), py::arg("kv_states"))
      .def(
          "update",
          [](KVStateCacheManager* self, std::vector<int>& prefix,
             std::vector<int>& tokens,
             const std::vector<std::vector<std::pair<LLMKV, LLMKV>>>&
                 kv_states) -> size_t {
            size_t updated = 0;
            VINEYARD_CHECK_OK(self->Update(prefix, tokens, kv_states, updated));
            return updated;
          },
          py::arg("prefix"), py::arg("tokens"), py::arg("kv_states"))
      .def(
          "query",
          [](KVStateCacheManager* self, std::vector<int>& tokens,
             int& next_token, std::vector<std::pair<LLMKV, LLMKV>>& kv_state) {
            VINEYARD_CHECK_OK(self->Query(tokens, next_token, kv_state));
          },
          py::arg("tokens"), py::arg("next_token"), py::arg("kv_states"))
      .def(
          "query",
          [](KVStateCacheManager* self, std::vector<int>& tokens,
             std::vector<std::vector<std::pair<LLMKV, LLMKV>>>& kv_state_list)
              -> size_t {
            size_t matched = 0;
            VINEYARD_CHECK_OK(self->Query(tokens, kv_state_list, matched));
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
             int split_number, std::string root, FilesystemType filesystemType)
              -> std::shared_ptr<KVStateCacheManager> {
            std::shared_ptr<KVStateCacheManager> manager;
            FileCacheConfig config(tensor_bytes, cache_capacity, layer,
                                   chunk_size, split_number, root,
                                   filesystemType);
            VINEYARD_CHECK_OK(
                vineyard::KVStateCacheManager::Make(manager, config));
            return manager;
          },
          py::arg("tensor_bytes") = 10, py::arg("cache_capacity") = 10,
          py::arg("layer") = 1, py::arg("chunk_size") = 5,
          py::arg("split_number") = 3, py::arg("root") = "root",
          py::arg("filesystem_type") = FilesystemType::LOCAL);
}

}  // namespace vineyard
