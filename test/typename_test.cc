/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "basic/ds/arrow.h"
#include "basic/ds/hashmap.h"
#include "common/util/logging.h"
#include "common/util/typename.h"

using namespace vineyard;  // NOLINT(build/namespaces)

template <typename K, typename V>
using my_hashmap = Hashmap<K, V>;

template <typename K, typename V>
struct T {};

template <typename K, typename V>
struct TXXXX {};

template <typename K, typename V>
using TX = TXXXX<K, V>;

int main(int, const char**) {
  {
    const auto type = type_name<int32_t>();
    CHECK_EQ(type, "int");
  }
  {
    const auto type = type_name<uint32_t>();
    CHECK_EQ(type, "uint");
  }
  {
    const auto type = type_name<int64_t>();
    CHECK_EQ(type, "int64");
  }
  {
    const auto type = type_name<uint64_t>();
    CHECK_EQ(type, "uint64");
  }
  {
    const auto type = type_name<std::string>();
    CHECK_EQ(type, "std::string");
  }
  {
    const auto type = type_name<Hashmap<int64_t, double>>();
    CHECK_EQ(type,
             "vineyard::Hashmap<int64,double,wy::hash<int64>,std::equal_to<"
             "int64>>");
  }

  {
    const auto type = type_name<Array<Hashmap<int64_t, double>::Entry>>();
    CHECK_EQ(type,
             "vineyard::Array<ska::detailv3::sherwood_v3_entry<std::pair<int64,"
             "double>>>");
  }

  {
    const auto type = type_name<Array<Hashmap<int64_t, uint64_t>::Entry>>();
    CHECK_EQ(type,
             "vineyard::Array<ska::detailv3::sherwood_v3_entry<std::pair<int64,"
             "uint64>>>");
  }

  {
    const auto type = type_name<my_hashmap<int64_t, double>>();
    CHECK_EQ(type,
             "vineyard::Hashmap<int64,double,wy::hash<int64>,std::equal_to<"
             "int64>>");
  }
  {
    const auto type = type_name<T<int64_t, std::string>>();
    CHECK_EQ(type, "T<int64,std::string>");
  }
  {
    const auto type = type_name<TXXXX<int64_t, std::string>>();
    CHECK_EQ(type, "TXXXX<int64,std::string>");
  }
  {
    const auto type =
        type_name<TX<int64_t, TXXXX<int64_t, T<int64_t, std::string>>>>();
    CHECK_EQ(type, "TXXXX<int64,TXXXX<int64,T<int64,std::string>>>");
  }
  {
    const auto type = type_name<std::hash<int>>();
    CHECK_EQ(type, "std::hash<int>");
  }
  {
    const auto type = type_name<wy::hash<int>>();
    CHECK_EQ(type, "wy::hash<int>");
  }
  {
    const auto type = type_name<std::equal_to<int64_t>>();
    CHECK_EQ(type, "std::equal_to<int64>");
  }
  {
    const auto type = type_name<vineyard::LargeStringArray>();
    CHECK_EQ(type, "vineyard::BaseBinaryArray<arrow::LargeStringArray>");
  }

  LOG(INFO) << "Passed typename tests...";

  return 0;
}
