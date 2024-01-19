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

#include "basic/ds/types.h"

namespace vineyard {

AnyType ParseAnyType(const std::string& type_name) {
  if (type_name == "int32") {
    return AnyType::Int32;
  } else if (type_name == "uint32") {
    return AnyType::UInt32;
  } else if (type_name == "int64") {
    return AnyType::Int64;
  } else if (type_name == "uint64") {
    return AnyType::UInt64;
  } else if (type_name == "float") {
    return AnyType::Float;
  } else if (type_name == "float64") {
    return AnyType::Double;
  } else if (type_name == "double") {
    return AnyType::Double;
  } else if (type_name == "string") {
    return AnyType::String;
  } else if (type_name == "date32") {
    return AnyType::Date32;
  } else if (type_name == "date64") {
    return AnyType::Date64;
  } else if (type_name == "time32") {
    return AnyType::Time32;
  } else if (type_name == "time64") {
    return AnyType::Time64;
  } else if (type_name == "timestamp") {
    return AnyType::Timestamp;
  } else {
    return AnyType::Undefined;
  }
}

std::string GetAnyTypeName(AnyType type) {
  switch (type) {
  case AnyType::Int32:
    return "int32";
  case AnyType::UInt32:
    return "uint32";
  case AnyType::Int64:
    return "int64";
  case AnyType::UInt64:
    return "uint64";
  case AnyType::Float:
    return "float";
  case AnyType::Double:
    return "double";
  case AnyType::String:
    return "string";
  case AnyType::Date32:
    return "date32";
  case AnyType::Date64:
    return "date64";
  case AnyType::Time32:
    return "time32";
  case AnyType::Time64:
    return "time64";
  case AnyType::Timestamp:
    return "timestamp";
  default:
    return "undefined";
  }
}

IdType ParseIdType(const std::string& type_name) {
  if (type_name == "int" || type_name == "int32" || type_name == "int32_t") {
    return IdType::Int32;
  } else if (type_name == "uint32" || type_name == "uint32_t") {
    return IdType::UInt32;
  } else if (type_name == "int64" || type_name == "int64_t") {
    return IdType::Int64;
  } else if (type_name == "uint64" || type_name == "uint64_t") {
    return IdType::UInt64;
  } else if (type_name == "string") {
    return IdType::String;
  } else if (type_name == "date32") {
    return IdType::Date32;
  } else if (type_name == "date64") {
    return IdType::Date64;
  } else if (type_name == "time32") {
    return IdType::Time32;
  } else if (type_name == "time64") {
    return IdType::Time64;
  } else if (type_name == "timestamp") {
    return IdType::Timestamp;
  } else {
    return IdType::Undefined;
  }
}

std::string GetIdTypeName(IdType type) {
  switch (type) {
  case IdType::Int32:
    return "int32";
  case IdType::UInt32:
    return "uint32";
  case IdType::Int64:
    return "int64";
  case IdType::UInt64:
    return "uint64";
  case IdType::String:
    return "string";
  case IdType::Date32:
    return "date32";
  case IdType::Date64:
    return "date64";
  case IdType::Time32:
    return "time32";
  case IdType::Time64:
    return "time64";
  case IdType::Timestamp:
    return "timestamp";
  default:
    return "undefined";
  }
}

std::ostream& operator<<(std::ostream& os, const AnyType& st) {
  os << GetAnyTypeName(st);
  return os;
}

std::istream& operator>>(std::istream& is, AnyType& st) {
  std::string name;
  is >> name;
  st = ParseAnyType(name);
  return is;
}

void to_json(json& j, const AnyType& type) { j = json(GetAnyTypeName(type)); }

void from_json(const json& j, AnyType& type) {
  type = ParseAnyType(j.get_ref<std::string const&>());
}

std::ostream& operator<<(std::ostream& os, const IdType& st) {
  os << GetIdTypeName(st);
  return os;
}

std::istream& operator>>(std::istream& is, IdType& st) {
  std::string name;
  is >> name;
  st = ParseIdType(name);
  return is;
}

void to_json(json& j, const IdType& type) { j = json(GetIdTypeName(type)); }

void from_json(const json& j, IdType& type) {
  type = ParseIdType(j.get_ref<std::string const&>());
}

}  // namespace vineyard
