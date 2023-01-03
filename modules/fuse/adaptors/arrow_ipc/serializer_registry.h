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

#ifndef MODULES_FUSE_ADAPTORS_ARROW_IPC_SERIALIZER_REGISTRY_H_
#define MODULES_FUSE_ADAPTORS_ARROW_IPC_SERIALIZER_REGISTRY_H_

#include <memory>
#include <string>

#include "basic/ds/dataframe.h"
#include "client/client.h"

namespace vineyard {
namespace fuse {

std::shared_ptr<arrow::Buffer> arrow_view(
    std::shared_ptr<vineyard::DataFrame>& df);

std::shared_ptr<arrow::Buffer> arrow_view(
    std::shared_ptr<vineyard::RecordBatch>& df);

std::shared_ptr<arrow::Buffer> arrow_view(std::shared_ptr<vineyard::Table>& df);

void from_arrow_view(Client* client, std::string const& name,
                     std::shared_ptr<arrow::BufferBuilder> buffer);

void from_arrow_view(Client* client, std::string const& path,
                     std::shared_ptr<arrow::Buffer> buffer);

}  // namespace fuse
}  // namespace vineyard

#endif  // MODULES_FUSE_ADAPTORS_ARROW_IPC_SERIALIZER_REGISTRY_H_
