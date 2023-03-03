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


GRIN_PARTITIONED_GRAPH get_partitioned_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id) {
  auto pg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(client.GetObject(object_id));
  return pg.get();
}

GRIN_GRAPH get_graph_by_object_id(vineyard::Client& client, const vineyard::ObjectID& object_id) {
  auto frag = std::dynamic_pointer_cast<vineyard::ArrowFragment<GRIN_OID_T, GRIN_VID_T>>(client.GetObject(object_id));
  return frag.get();
}

std::string GetDataTypeName(GRIN_DATATYPE type) {
  switch (type) {
  case GRIN_DATATYPE::Int32:
    return "int32";
  case GRIN_DATATYPE::UInt32:
    return "uint32";
  case GRIN_DATATYPE::Int64:
    return "int64";
  case GRIN_DATATYPE::UInt64:
    return "uint64";
  case GRIN_DATATYPE::Float:
    return "float";
  case GRIN_DATATYPE::Double:
    return "double";
  case GRIN_DATATYPE::String:
    return "string";
  case GRIN_DATATYPE::Date32:
    return "date32";
  case GRIN_DATATYPE::Date64:
    return "date64";
  default:
    return "undefined";
  }
}

GRIN_DATATYPE ArrowToDataType(std::shared_ptr<arrow::DataType> type) {
  if (type == nullptr) {
    return GRIN_DATATYPE::Undefined;
  } else if (arrow::int32()->Equals(type)) {
    return GRIN_DATATYPE::Int32;
  } else if (arrow::int64()->Equals(type)) {
    return GRIN_DATATYPE::Int64;
  } else if (arrow::float32()->Equals(type)) {
    return GRIN_DATATYPE::Float;
  } else if (arrow::uint32()->Equals(type)) {
    return GRIN_DATATYPE::UInt32;
  } else if (arrow::uint64()->Equals(type)) {
    return GRIN_DATATYPE::UInt64;
  } else if (arrow::float64()->Equals(type)) {
    return GRIN_DATATYPE::Double;
  } else if (arrow::utf8()->Equals(type)) {
    return GRIN_DATATYPE::String;
  } else if (arrow::large_utf8()->Equals(type)) {
    return GRIN_DATATYPE::String;
  } 
  return GRIN_DATATYPE::Undefined;
}
