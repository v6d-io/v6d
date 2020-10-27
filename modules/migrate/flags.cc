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

#include "migrate/flags.h"

namespace vineyard {

DEFINE_uint64(migration_port, 0, "rpc port of migration");
DEFINE_string(object_list, "", "object list");
DEFINE_string(instance_map, "", "instance_mapping");
DEFINE_string(ipc_socket, "", "ipc socket of vineyard server");

}  // namespace vineyard
