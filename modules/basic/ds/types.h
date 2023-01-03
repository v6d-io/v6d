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

#ifndef MODULES_BASIC_DS_TYPES_H_
#define MODULES_BASIC_DS_TYPES_H_

#include <stdint.h>

#include <iostream>
#include <string>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "common/util/json.h"

namespace vineyard {

enum class AnyType {
  Undefined = 0,
  Int32 = 1,
  UInt32 = 2,
  Int64 = 3,
  UInt64 = 4,
  Float = 5,
  Double = 6,
  String = 7,
  Date32 = 8,
  Date64 = 9,
};

template <typename T>
struct AnyTypeEnum {
  static constexpr AnyType value = AnyType::Undefined;
};

template <>
struct AnyTypeEnum<int32_t> {
  static constexpr AnyType value = AnyType::Int32;
};

template <>
struct AnyTypeEnum<uint32_t> {
  static constexpr AnyType value = AnyType::UInt32;
};

template <>
struct AnyTypeEnum<int64_t> {
  static constexpr AnyType value = AnyType::Int64;
};

template <>
struct AnyTypeEnum<uint64_t> {
  static constexpr AnyType value = AnyType::UInt64;
};

template <>
struct AnyTypeEnum<float> {
  static constexpr AnyType value = AnyType::Float;
};

template <>
struct AnyTypeEnum<double> {
  static constexpr AnyType value = AnyType::Double;
};

template <>
struct AnyTypeEnum<std::string> {
  static constexpr AnyType value = AnyType::String;
};

template <>
struct AnyTypeEnum<arrow::Date32Type> {
  static constexpr AnyType value = AnyType::Date32;
};

template <>
struct AnyTypeEnum<arrow::Date64Type> {
  static constexpr AnyType value = AnyType::Date64;
};

enum class IdType {
  Undefined = 0,
  Int32 = 1,
  Int64 = 2,
  UInt32 = 3,
  UInt64 = 4,
  String = 5,
  Date32 = 6,
  Date64 = 7,
};

template <typename T>
struct IdTypeEnum {
  static constexpr IdType value = IdType::Undefined;
};

template <>
struct IdTypeEnum<int32_t> {
  static constexpr IdType value = IdType::Int32;
};

template <>
struct IdTypeEnum<int64_t> {
  static constexpr IdType value = IdType::Int64;
};

template <>
struct IdTypeEnum<uint32_t> {
  static constexpr IdType value = IdType::UInt32;
};

template <>
struct IdTypeEnum<uint64_t> {
  static constexpr IdType value = IdType::UInt64;
};

template <>
struct IdTypeEnum<std::string> {
  static constexpr IdType value = IdType::String;
};

template <>
struct IdTypeEnum<arrow::Date32Type> {
  static constexpr IdType value = IdType::Date32;
};

template <>
struct IdTypeEnum<arrow::Date64Type> {
  static constexpr IdType value = IdType::Date64;
};

AnyType ParseAnyType(const std::string& type_name);

std::string GetAnyTypeName(AnyType type);

IdType ParseIdType(const std::string& type_name);

std::string GetIdTypeName(IdType type);

std::ostream& operator<<(std::ostream& os, const AnyType& st);

std::istream& operator>>(std::istream& is, AnyType& st);

void to_json(json& j, const AnyType& type);

void from_json(const json& j, AnyType& type);

std::ostream& operator<<(std::ostream& os, const IdType& st);

std::istream& operator>>(std::istream& is, IdType& st);

void to_json(json& j, const IdType& type);

void from_json(const json& j, IdType& type);

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_TYPES_H_
