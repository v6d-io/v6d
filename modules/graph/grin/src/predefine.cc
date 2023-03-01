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

#include "graph/grin/src/predefine.h"


PartitionedGraph get_partitioned_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id) {
  auto pg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(client.GetObject(object_id));
  return pg.get();
}

Graph get_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id) {
  auto frag = std::dynamic_pointer_cast<vineyard::ArrowFragment<G_OID_T, G_VID_T>>(client.GetObject(object_id));
  return frag.get();
}

std::string GetDataTypeName(DataType type) {
  switch (type) {
  case DataType::Int32:
    return "int32";
  case DataType::UInt32:
    return "uint32";
  case DataType::Int64:
    return "int64";
  case DataType::UInt64:
    return "uint64";
  case DataType::Float:
    return "float";
  case DataType::Double:
    return "double";
  case DataType::String:
    return "string";
  case DataType::Date32:
    return "date32";
  case DataType::Date64:
    return "date64";
  default:
    return "undefined";
  }
}

DataType ArrowToDataType(std::shared_ptr<arrow::DataType> type) {
  if (type == nullptr) {
    return DataType::Undefined;
  } else if (arrow::int32()->Equals(type)) {
    return DataType::Int32;
  } else if (arrow::int64()->Equals(type)) {
    return DataType::Int64;
  } else if (arrow::float32()->Equals(type)) {
    return DataType::Float;
  } else if (arrow::uint32()->Equals(type)) {
    return DataType::UInt32;
  } else if (arrow::uint64()->Equals(type)) {
    return DataType::UInt64;
  } else if (arrow::float64()->Equals(type)) {
    return DataType::Double;
  } else if (arrow::utf8()->Equals(type)) {
    return DataType::String;
  } else if (arrow::large_utf8()->Equals(type)) {
    return DataType::String;
  } 
  return DataType::Undefined;
}
