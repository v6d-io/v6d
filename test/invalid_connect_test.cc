/** Copyright 2021 Alibaba Group Holding Limited.

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

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/backtrace/backtrace.hpp"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./invalid_connect_test <ipc_socket> <rpc_endpoint>");
    return 1;
  }
  std::string ipc_socket(argv[1]);
  std::string rpc_endpoint(argv[2]);

  // valid

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  LOG(INFO) << "ipc socket is: " << client.IPCSocket();
  LOG(INFO) << "rpc endpoint is: " << client.RPCEndpoint();

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
  LOG(INFO) << "Connected to RPCServer: " << rpc_endpoint;

  LOG(INFO) << "ipc socket is: " << client.IPCSocket();
  LOG(INFO) << "rpc endpoint is: " << client.RPCEndpoint();

  // invalid

  {
    Client invalid;
    auto s1 = client.Connect("/connect/to/some/where");
    LOG(INFO) << s1.ToString();
    CHECK(s1.IsAssertionFailed() || s1.IsIOError() || s1.IsConnectionFailed());
  }

  {
    RPCClient invalid;
    auto s1 = client.Connect("/connect/to/some/where");
    LOG(INFO) << s1.ToString();
    CHECK(s1.IsAssertionFailed() || s1.IsIOError() || s1.IsConnectionFailed());
  }

  {
    RPCClient invalid;
    auto s1 = client.Connect("invalid-host.com");
    LOG(INFO) << s1.ToString();
    CHECK(s1.IsAssertionFailed() || s1.IsIOError() || s1.IsConnectionFailed());
  }

  // test libunwind and backtrace
  std::stringstream ss;
  backtrace_info::backtrace(ss, true);
  LOG(INFO) << ss.str();
  ss.str("");
  backtrace_info::backtrace(ss, false);
  LOG(INFO) << ss.str();
  ss.str("");

  backtrace_info::backtrace(std::cout, true);
  backtrace_info::backtrace(std::cout, false);

  LOG(INFO) << "Passed invalid connect tests...";

  client.Disconnect();

  return 0;
}
