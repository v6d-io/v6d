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
#include "grin/include/predefine.h"
#include "src/client/client.h"
#include "modules/graph/fragment/arrow_fragment.h"


#ifndef GRIN_SRC_UTILS_H_
#define GRIN_SRC_UTILS_H_

Graph get_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id) {
    auto frag = std::dynamic_pointer_cast<vineyard::ArrowFragment<uint32_t, uint32_t>>(client.GetObject(object_id));
    return frag.get();
}

#endif // GRIN_SRC_UTILS_H_