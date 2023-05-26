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

#include "graph/fragment/arrow_fragment_builder_impl.h"

namespace vineyard {

template class ArrowFragment<
    int64_t, uint64_t,
    ArrowVertexMap<typename InternalType<int64_t>::type, uint64_t>, false>;

template class ArrowFragment<
    int64_t, uint64_t,
    ArrowLocalVertexMap<typename InternalType<int64_t>::type, uint64_t>, false>;

template class BasicArrowFragmentBuilder<
    int64_t, uint64_t,
    ArrowVertexMap<typename InternalType<int64_t>::type, uint64_t>, false>;

template class BasicArrowFragmentBuilder<
    int64_t, uint64_t,
    ArrowLocalVertexMap<typename InternalType<int64_t>::type, uint64_t>, false>;

template class ArrowFragment<
    int64_t, uint64_t,
    ArrowVertexMap<typename InternalType<int64_t>::type, uint64_t>, true>;

template class ArrowFragment<
    int64_t, uint64_t,
    ArrowLocalVertexMap<typename InternalType<int64_t>::type, uint64_t>, true>;

template class BasicArrowFragmentBuilder<
    int64_t, uint64_t,
    ArrowVertexMap<typename InternalType<int64_t>::type, uint64_t>, true>;

template class BasicArrowFragmentBuilder<
    int64_t, uint64_t,
    ArrowLocalVertexMap<typename InternalType<int64_t>::type, uint64_t>, true>;

}  // namespace vineyard
