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
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "client/client.h"
#include "common/util/env.h"
#include "common/util/logging.h"

#if defined(WITH_HOSSEINMOEIN_DATAFRAME)
#include "DataFrame/DataFrame.h"
#include "hosseinmoein-dataframe/hosseinmoein_dataframe.h"
#endif

using namespace vineyard;  // NOLINT(build/namespaces)

#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_WHITE "\033[37m"

#define CUR_BACK "\033[1D"

#define MAX_ROW (20)
#define MAX_DATA (9973)

#define OFF 0
#define ON 1

int row;

template <typename T>
std::ostream& operator<<(std::ostream& o, std::vector<T>& vec) {
  o << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    o << " " << vec[i] << ",";
  }
  o << CUR_BACK << " ]";
  return o;
}

void init() {
  srand(time(0));
  row = rand() % (MAX_ROW - 1) + 1;  // NOLINT(runtime/threadsafe_fn)
}

template <typename T>
void fill_random_data(std::vector<T>& vec) {
  for (int i = 0; i < row; i++) {
    vec[i] = ((T)(rand() % MAX_DATA + 1)) / 3;  // NOLINT(runtime/threadsafe_fn)
  }
}

template <typename T>
int test_func(Client& client) {
#if defined(WITH_HOSSEINMOEIN_DATAFRAME)
  auto builder = HDataFrameBuilder<T>();
  StdDataFrame<T> data_frame_input;
  StdDataFrame<T> data_frame_output;

  /* Index and column data.*/
  std::vector<T> idx_col(row), idx_col_result(row);
  std::vector<int32_t> int32_col(row), int32_col_result(row);
  std::vector<int64_t> int64_col(row), int64_col_result(row);
  std::vector<uint32_t> uint32_col(row), uint32_col_result(row);
  std::vector<uint64_t> uint64_col(row), uint64_col_result(row);
  std::vector<double> double_col(row), double_col_result(row);
  std::vector<float> float_col(row), float_col_result(row);

  fill_random_data(idx_col);
  fill_random_data(int32_col);
  fill_random_data(int64_col);
  fill_random_data(uint32_col);
  fill_random_data(uint64_col);
  fill_random_data(double_col);
  fill_random_data(float_col);

  DLOG(INFO) << "Write:";
  DLOG(INFO) << "idx_col:" << std::endl << idx_col;
  DLOG(INFO) << "int32_col:" << std::endl << int32_col;
  DLOG(INFO) << "int64_col:" << std::endl << int64_col;
  DLOG(INFO) << "uint32_col:" << std::endl << uint32_col;
  DLOG(INFO) << "uint64_col:" << std::endl << uint64_col;
  DLOG(INFO) << "double_col:" << std::endl << double_col;
  DLOG(INFO) << "float_col:" << std::endl << float_col;

  data_frame_input.load_data(std::move(idx_col),
                             std::make_pair("int32_col", int32_col),
                             std::make_pair("int64_col", int64_col),
                             std::make_pair("uint32_col", uint32_col),
                             std::make_pair("uint64_col", uint64_col),
                             std::make_pair("double_col", double_col),
                             std::make_pair("float_col", float_col));

  builder.Put(data_frame_input);
  auto result = builder.Seal(client);

  auto hdf =
      std::dynamic_pointer_cast<HDataFrame<T>>(client.GetObject(result->id()));
  data_frame_output = hdf->Resolve(client);

  idx_col_result = data_frame_output.get_index();
  int32_col_result =
      data_frame_output.template get_column<int32_t>("int32_col");
  int64_col_result =
      data_frame_output.template get_column<int64_t>("int64_col");
  uint32_col_result =
      data_frame_output.template get_column<uint32_t>("uint32_col");
  uint64_col_result =
      data_frame_output.template get_column<uint64_t>("uint64_col");
  double_col_result =
      data_frame_output.template get_column<double>("double_col");
  float_col_result = data_frame_output.template get_column<float>("float_col");

  DLOG(INFO) << "Read:";
  DLOG(INFO) << "idx_col:" << std::endl << idx_col_result;
  DLOG(INFO) << "int32_col:" << std::endl << int32_col_result;
  DLOG(INFO) << "int64_col:" << std::endl << int64_col_result;
  DLOG(INFO) << "uint32_col:" << std::endl << uint32_col_result;
  DLOG(INFO) << "uint64_col:" << std::endl << uint64_col_result;
  DLOG(INFO) << "double_col:" << std::endl << double_col_result;
  DLOG(INFO) << "float_col:" << std::endl << float_col_result;

  for (int i = 0; i < row; i++) {
    if (idx_col_result[i] != idx_col[i] ||
        int32_col_result[i] != int32_col[i] ||
        int64_col_result[i] != int64_col[i] ||
        uint32_col_result[i] != uint32_col[i] ||
        uint64_col_result[i] != uint64_col[i] ||
        double_col_result[i] != double_col[i] ||
        float_col_result[i] != float_col[i]) {
      return -1;
    }
  }

#endif  // WITH_HOSSEINMOEIN_DATAFRAME
  return 0;
}

int test_different_types(Client& client) {
  {
    int ret = test_func<int32_t>(client);
    if (ret) {
      LOG(INFO) << "Failed case index type: int32_t";
      return ret;
    }
  }

  {
    int ret = test_func<int64_t>(client);
    if (ret) {
      LOG(INFO) << "Failed case index type: int64_t";
      return ret;
    }
  }

  {
    int ret = test_func<uint32_t>(client);
    if (ret) {
      LOG(INFO) << "Failed case index type: uint32_t";
      return ret;
    }
  }

  {
    int ret = test_func<uint64_t>(client);
    if (ret) {
      LOG(INFO) << "Failed case index type: uint64_t";
      return ret;
    }
  }

  {
    int ret = test_func<double>(client);
    if (ret) {
      LOG(INFO) << "Failed case index type: double";
      return ret;
    }
  }

  {
    int ret = test_func<float>(client);
    if (ret) {
      LOG(INFO) << "Failed case index type: float";
      return ret;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./hosseinmoein_dataframe_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  init();

  int ret = test_different_types(client);
  if (ret) {
    LOG(INFO) << "Hosseinmoein dataframe test:" << COLOR_RED << " FAIL"
              << COLOR_WHITE;
  } else {
    LOG(INFO) << "Hosseinmoein dataframe test:" << COLOR_GREEN << " PASS"
              << COLOR_WHITE;
  }

  client.Disconnect();
  return ret;
}
