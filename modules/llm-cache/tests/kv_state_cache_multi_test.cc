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

#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "common/util/logging.h"

constexpr const char* program = "./build/bin/kv_state_cache_test";

pid_t create_subprocess(char* argv[]) {
  pid_t pid = fork();
  if (pid == -1) {
    std::cerr << "Failed to fork()" << std::endl;
    exit(1);
  } else if (pid > 0) {
    return pid;
  } else {
    execv(program, argv);
    std::cerr << "Failed to exec()" << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  char process_name[] = "kv_state_cache_test";
  char arg_0[] = "-s";
  char token_sequence_1[] = "1";
  char token_sequence_2[] = "2";
  char token_sequence_3[] = "3";
  char token_sequence_4[] = "4";

  std::string sockets[2];
  std::string rpc_endpoint;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--vineyard-endpoint") == 0) {
      rpc_endpoint = std::string(argv[i + 1]);
    } else if (strcmp(argv[i], "--vineyard-ipc-sockets") == 0) {
      sockets[0] = std::string(argv[i + 1]);
      sockets[1] = std::string(argv[i + 2]);
    }
  }

  char* socket_str[2];
  socket_str[0] =
      (char*) malloc(sockets[0].length() + 1);  // NOLINT(readability/casting)
  socket_str[1] =
      (char*) malloc(sockets[1].length() + 1);  // NOLINT(readability/casting)
  memset(socket_str[0], 0, sockets[0].length() + 1);
  memset(socket_str[1], 0, sockets[1].length() + 1);
  strcpy(socket_str[0], sockets[0].c_str());  // NOLINT(runtime/printf)
  strcpy(socket_str[1], sockets[1].c_str());  // NOLINT(runtime/printf)

  char* args_1[] = {process_name,     socket_str[0],    arg_0,
                    token_sequence_1, token_sequence_2, token_sequence_3,
                    token_sequence_4, nullptr};
  char* args_2[] = {process_name,     socket_str[1],    arg_0,
                    token_sequence_1, token_sequence_2, token_sequence_3,
                    nullptr};

  std::vector<pid_t> pids;
  pids.push_back(create_subprocess(args_1));
  pids.push_back(create_subprocess(args_2));
  for (size_t i = 0; i < pids.size(); i++) {
    int status;
    waitpid(pids[i], &status, 0);
    if ((!WIFEXITED(status)) || WEXITSTATUS(status) != 0) {
      free(socket_str[0]);
      free(socket_str[1]);
      LOG(INFO) << "child error!";
      return 1;
    }
  }

  free(socket_str[0]);
  free(socket_str[1]);

  return 0;
}
