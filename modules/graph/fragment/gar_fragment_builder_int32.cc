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

#ifdef ENABLE_GAR

#include "graph/fragment/gar_fragment_builder_impl.h"

namespace vineyard {

template class GARFragmentBuilder<
    int32_t, uint32_t,
    ArrowVertexMap<typename InternalType<int32_t>::type, uint32_t>>;

template class GARFragmentBuilder<
    int32_t, uint64_t,
    ArrowVertexMap<typename InternalType<int32_t>::type, uint64_t>>;

}  // namespace vineyard

#endif  // ENABLE_GAR
