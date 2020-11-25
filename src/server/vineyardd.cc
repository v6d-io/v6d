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

#include <signal.h>

#include <iostream>

#include "common/backtrace/backtrace.hpp"
#include "common/util/flags.h"
#include "common/util/logging.h"
#include "server/server/vineyard_server.h"
#include "server/util/spec_resolvers.h"

DECLARE_bool(help);
DECLARE_string(helpmatch);

// we need a global reference of the server_ptr to do cleanup
static std::shared_ptr<vineyard::VineyardServer> server_ptr_ = nullptr;

void terminate_handle() {
#ifndef WITHOUT_LIBUNWIND
  backtrace(std::cerr);
#endif
}

extern "C" void vineyardd_signal_handler(int sig) {
  if (sig == SIGTERM || sig == SIGINT) {
    // exit normally to guarantee resources are released correctly via dtor.
    if (server_ptr_) {
      LOG(INFO) << "SIGTERM Signal received, stop vineyard server...";
      server_ptr_->Stop();
    }
  }
}

int main(int argc, char* argv[]) {
  // restore the default signal handling, when "vineyardd" is launched inside
  // a bash script.
  sigset(SIGINT, SIG_DFL);
  FLAGS_stderrthreshold = 0;
  vineyard::logging::InitGoogleLogging("vineyard");
  vineyard::logging::InstallFailureSignalHandler();
  std::set_terminate(terminate_handle);

  vineyard::flags::SetUsageMessage("Usage: vineyardd [options]");
  vineyard::flags::ParseCommandLineNonHelpFlags(&argc, &argv, false);
  if (FLAGS_help) {
    FLAGS_help = false;
    FLAGS_helpmatch = "vineyard";
  }
  vineyard::flags::HandleCommandLineHelpFlags();

  LOG(INFO) << "Hello vineyard!";

  // Ignore SIGPIPE signals to avoid killing the server when writting to a lost
  // client connection.
  signal(SIGPIPE, SIG_IGN);

  const auto& spec = vineyard::ServerSpecResolver().resolve();
  server_ptr_ = vineyard::VineyardServer::Get(spec);
  // do proper cleanup in response to signals
  signal(SIGINT, vineyardd_signal_handler);
  signal(SIGTERM, vineyardd_signal_handler);
  VINEYARD_CHECK_OK(server_ptr_->Serve());
  VINEYARD_CHECK_OK(server_ptr_->Finalize());

  vineyard::flags::ShutDownCommandLineFlags();
  vineyard::logging::ShutdownGoogleLogging();
  return 0;
}
