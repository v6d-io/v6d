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

#include "llm-cache/ds/kv_state_cache_manager.h"

namespace py = pybind11;

namespace vineyard {

PYBIND11_MODULE(llm_C, m) {
    m.doc() = "vineyard llm kv cache manager module";
    
    py::class_<KVStateCacheManager>(m, "KVStateCacheManager")
        .def(py::init<vineyard::Client&, std::shared_ptr<vineyard::KVStateCacheBuilder>&, int, std::string&, std::string&>())
        .def("Make", &vineyard::KVStateCacheManager::Make);
}

}  // namespace vineyard
