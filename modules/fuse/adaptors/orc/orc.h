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

#ifndef MODULES_FUSE_ADAPTORS_ORC_ORC_H_
#define MODULES_FUSE_ADAPTORS_ORC_ORC_H_

#if defined(WITH_ORC)

#include <memory>

#include "basic/ds/dataframe.h"
#include "client/client.h"

namespace vineyard {
namespace fuse {

void orc_view(std::shared_ptr<vineyard::DataFrame>& df);

}  // namespace fuse
}  // namespace vineyard

#endif

#endif  // MODULES_FUSE_ADAPTORS_ORC_ORC_H_
