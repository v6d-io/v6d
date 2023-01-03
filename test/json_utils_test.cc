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

#include <memory>
#include <string>
#include <thread>

#include "basic/ds/types.h"
#include "common/util/json.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  json tree;
  auto value1 = std::vector<int>{1, 2, 3};
  auto value2 = std::vector<double>{1.0, 2.0, 3.0};
  auto value3 = std::vector<std::string>{"a", "bb", "ccc"};
  put_container(tree, "value1", value1);
  put_container(tree, "value2", value2);
  put_container(tree, "value3", value3);

  std::vector<int> value1_get;
  std::vector<double> value2_get;
  std::vector<std::string> value3_get;
  get_container(tree, "value1", value1_get);
  get_container(tree, "value2", value2_get);
  get_container(tree, "value3", value3_get);
  CHECK(value1 == value1_get);
  CHECK(value2 == value2_get);
  CHECK(value3 == value3_get);

  LOG(INFO) << "Passed plain type in json tests...";

  auto value4 =
      std::vector<AnyType>{AnyType::Int32, AnyType::UInt32, AnyType::Double};
  put_container(tree, "value4", value4);
  std::vector<AnyType> value4_get;
  get_container(tree, "value4", value4_get);
  CHECK(value4 == value4_get);

  LOG(INFO) << "Passed AnyType in json tests...";

  auto value5 =
      std::vector<IdType>{IdType::Int32, IdType::UInt32, IdType::String};
  put_container(tree, "value5", value5);
  std::vector<IdType> value5_get;
  get_container(tree, "value5", value5_get);
  CHECK(value5 == value5_get);

  LOG(INFO) << "Passed IdType in json tests...";

  return 0;
}
