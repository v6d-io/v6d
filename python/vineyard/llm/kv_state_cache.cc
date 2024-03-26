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

LLMKV create_llmkv_from_buffer(py::buffer buffer, size_t size) {
  py::buffer_info info = buffer.request();
  return LLMKV{info.ptr, size};
}

PYBIND11_MODULE(llm_C, m) {
  m.doc() = "vineyard llm kv cache manager module";

  pybind11::enum_<FilesystemType>(m, "FilesystemType")
      .value("LOCAL", FilesystemType::LOCAL)
      .export_values();

  py::class_<LLMKV>(m, "KVTensor")
      .def(py::init(&create_llmkv_from_buffer), py::arg("buffer"),
           py::arg("size"))
      .def_readwrite("length", &LLMKV::length)
      .def("data", [](LLMKV& self) -> py::buffer_info {
        return py::buffer_info(self.data, sizeof(char),
                               py::format_descriptor<char>::value, 1,
                               {self.length}, {sizeof(char)});
      });

  py::class_<KVStateCacheManager, std::shared_ptr<KVStateCacheManager>>(
      m, "KVStateCacheManager")
      .def(
          "update",
          [](KVStateCacheManager* self, std::vector<int>& tokenList,
             int& nextToken,
             const std::map<int, std::pair<vineyard::LLMKV, vineyard::LLMKV>>&
                 kv_state) {
            VINEYARD_CHECK_OK(self->Update(tokenList, nextToken, kv_state));
          },
          py::arg("tokenList"), py::arg("nextToken"), py::arg("kv_state"))
      .def(
          "update",
          [](KVStateCacheManager* self, std::vector<int>& tokenList,
             const std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>&
                 kv_state_list) {
            VINEYARD_CHECK_OK(self->Update(tokenList, kv_state_list));
          },
          py::arg("tokenList"), py::arg("kv_state_list"))
      .def(
          "query",
          [](KVStateCacheManager* self, std::vector<int>& tokenList,
             int& nextToken, std::map<int, std::pair<LLMKV, LLMKV>>& kv_state) {
            VINEYARD_CHECK_OK(self->Query(tokenList, nextToken, kv_state));
          },
          py::arg("tokenList"), py::arg("nextToken"), py::arg("kv_state"))
      .def(
          "query",
          [](KVStateCacheManager* self, std::vector<int>& tokenList,
             std::vector<std::map<int, std::pair<LLMKV, LLMKV>>>&
                 kv_state_list) {
            VINEYARD_CHECK_OK(self->Query(tokenList, kv_state_list));
          },
          py::arg("tokenList"), py::arg("kv_state_list"))
      .def("close", [](KVStateCacheManager* self) { self->Close(); });

  m.def(
       "_generate",
       [](py::object ipc_client, int tensor_bytes, int cache_capacity,
          int layer, int block_size, int sync_interval,
          std::string llm_cache_sync_lock, std::string llm_cache_object_name,
          std::string llm_ref_cnt_object_name) -> py::object {
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
         return py::cast(manager);
       },
       py::arg("ipc_client"), py::arg("tensor_bytes") = 10,
       py::arg("cache_capacity") = 10, py::arg("layer") = 1,
       py::arg("block_size") = 5, py::arg("sync_interval") = 3,
       py::arg("llm_cache_sync_lock") = "llmCacheSyncLock",
       py::arg("llm_cache_object_name") = "llm_cache_object",
       py::arg("llm_ref_cnt_object_name") = "llm_refcnt_object")
      .def(
          "_generate",
          [](int tensor_bytes, int cache_capacity, int layer, int batch_size,
             int split_number, std::string root,
             FilesystemType filesystemType) -> py::object {
            std::shared_ptr<KVStateCacheManager> manager;
            FileCacheConfig config(tensor_bytes, cache_capacity, layer,
                                   batch_size, split_number, root,
                                   filesystemType);
            vineyard::Status status =
                vineyard::KVStateCacheManager::Make(manager, config);
            if (!status.ok()) {
              throw std::runtime_error(status.ToString());
            }
            return py::cast(manager);
          },
          py::arg("tensor_bytes") = 10, py::arg("cache_capacity") = 10,
          py::arg("layer") = 1, py::arg("batch_size") = 5,
          py::arg("split_number") = 3, py::arg("root") = "root",
          py::arg("filesystem_type") = FilesystemType::LOCAL);
}

}  // namespace vineyard
