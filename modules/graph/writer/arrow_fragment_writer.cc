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

#include "graph/writer/arrow_fragment_writer.h"

#include <memory>
#include <set>
#include <vector>

#include "arrow/api.h"

namespace vineyard {

void FinishArrowArrayBuilders(
    std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders,
    std::vector<std::shared_ptr<arrow::Array>>& columns) {
  for (size_t i = 0; i < builders.size(); i++) {
    ARROW_CHECK_OK(builders[i]->Finish(&columns[i]));
  }
}

void ResetArrowArrayBuilders(
    std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders) {
  for (size_t i = 0; i < builders.size(); i++) {
    builders[i]->Reset();
  }
}

void InitializeArrayArrayBuilders(
    std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders,
    const std::set<property_graph_types::LABEL_ID_TYPE>& property_ids,
    const property_graph_types::LABEL_ID_TYPE edge_label,
    const PropertyGraphSchema& graph_schema) {
  builders.resize(property_ids.size() + 2);
  builders[0] = std::make_shared<arrow::Int64Builder>();  // vertex index column
  builders[1] = std::make_shared<arrow::Int64Builder>();  // vertex index column
  int col_id = 2;
  for (auto& pid : property_ids) {
    auto prop_type = graph_schema.GetEdgePropertyType(edge_label, pid);
    if (arrow::boolean()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::BooleanBuilder>();
    } else if (arrow::int32()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::Int32Builder>();
    } else if (arrow::int64()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::Int64Builder>();
    } else if (arrow::float32()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::FloatBuilder>();
    } else if (arrow::float64()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::DoubleBuilder>();
    } else if (arrow::utf8()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::StringBuilder>();
    } else if (arrow::large_utf8()->Equals(prop_type)) {
      builders[col_id] = std::make_shared<arrow::LargeStringBuilder>();
    }
    ++col_id;
  }
}

}  // namespace vineyard

#endif  // ENABLE_GAR
