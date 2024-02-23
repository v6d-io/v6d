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

#include "graph/loader/gar_fragment_loader.h"

#include "gar/util/data_type.h"

#include "arrow/api.h"
#include "gar/graph_info.h"

namespace vineyard {

std::shared_ptr<arrow::Schema> ConstructSchemaFromPropertyGroup(
    const std::shared_ptr<GraphArchive::PropertyGroup>& property_group) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& prop : property_group->GetProperties()) {
    fields.emplace_back(arrow::field(
        prop.name, GraphArchive::DataType::DataTypeToArrowDataType(prop.type)));
  }
  return arrow::schema(fields);
}

}  // namespace vineyard

#endif  // ENABLE_GAR
