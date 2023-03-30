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

namespace py = pybind11;

namespace vineyard {

void bind_error(py::module& mod);
void bind_core(py::module& mod);
void bind_blobs(py::module& mod);
void bind_client(py::module& mod);
void bind_utils(py::module& mod);
void bind_stream(py::module& mod);

PYBIND11_MODULE(_C, mod) {
  py::options options;
  options.enable_user_defined_docstrings();
  options.disable_function_signatures();

  bind_error(mod);
  bind_core(mod);
  bind_blobs(mod);
  bind_client(mod);
  bind_utils(mod);

#if defined(BIND_STREAM)
  bind_stream(mod);
#endif
}

}  // namespace vineyard
