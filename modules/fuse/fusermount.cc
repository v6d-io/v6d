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

#include <stdio.h>

#include <limits>
#include <memory>
#include <string>
#include <thread>

#include "adaptors/arrow_ipc/deserializer_registry.h"
#include "common/util/env.h"
#include "common/util/logging.h"
#include "fuse/fuse_impl.h"

/*
 * Command line options
 *
 * Don't set default values for the char* fields here because fuse_opt_parse
 * would attempt to free() them when the user specifies different values on the
 * command line.
 */
static struct options {
  const char* vineyard_socket;
  int show_help;
} options;

#define OPTION(t, p) \
  { t, offsetof(struct options, p), 1 }

static const struct fuse_opt option_spec[] = {
    OPTION("--vineyard-socket=%s", vineyard_socket),
    OPTION("--help", show_help), OPTION("-h", show_help), FUSE_OPT_END};

static void print_help(const char* progname) {
  printf("usage: %s [options] <mountpoint>\n\n", progname);
  printf(
      "Vineyard specific options:\n"
      "    --vineyard-socket=<s>  Path of UNIX-domain socket of vineyard "
      "server\n"
      "                           (default: \"$VINEYARD_IPC_SOCKET\")\n"
      "\n");
}

static int process_args(struct fuse_args& args, int argc, char** argv) {
  // Set defaults -- we have to use strdup so that fuse_opt_parse can free
  // the defaults if other values are specified.
  if (!options.vineyard_socket) {
    std::string env = vineyard::read_env("VINEYARD_IPC_SOCKET");

    options.vineyard_socket = strdup(env.c_str());
  }

  /* Parse options */
  if (fuse_opt_parse(&args, &options, option_spec, NULL) == -1) {
    LOG(ERROR) << "Failed to parse command line options.";
    print_help(argv[0]);
    return 1;
  }

  /* When --help is specified, first print our own file-system
     specific help text, then signal fuse_main to show
     additional help (by adding `--help` to the options again)
     without usage: line (by setting argv[0] to the empty
     string) */
  if (options.show_help) {
    print_help(argv[0]);
    assert(fuse_opt_add_arg(&args, "--help") == 0);
    args.argv[0][0] = '\0';
  }

  // force running as foreground mode
  assert(fuse_opt_add_arg(&args, "-f") == 0);

  // populate state
  vineyard::fuse::fs::state.vineyard_socket = options.vineyard_socket;
  LOG(INFO) << "prepare to conncet to socket"
            << vineyard::fuse::fs::state.vineyard_socket;

  vineyard::fuse::fs::state.ipc_desearilizer_registry =
      vineyard::fuse::arrow_ipc_register_once();
  return 0;
}

static const struct fuse_operations vineyard_fuse_operations = {
    .getattr = vineyard::fuse::fs::fuse_getattr,
    .open = vineyard::fuse::fs::fuse_open,
    .read = vineyard::fuse::fs::fuse_read,
    .write = vineyard::fuse::fs::fuse_write,
    .statfs = vineyard::fuse::fs::fuse_statfs,
    .flush = vineyard::fuse::fs::fuse_flush,
    .release = vineyard::fuse::fs::fuse_release,
    .getxattr = vineyard::fuse::fs::fuse_getxattr,
    .opendir = vineyard::fuse::fs::fuse_opendir,
    .readdir = vineyard::fuse::fs::fuse_readdir,
    .init = vineyard::fuse::fs::fuse_init,
    .destroy = vineyard::fuse::fs::fuse_destroy,
    .create = vineyard::fuse::fs::fuse_create,

    // .access = vineyard::fuse::fs::fuse_access,
};

int main(int argc, char* argv[]) {
  // restore the default signal handling, when "vineyard-fusermount" is
  // launched inside a bash script.
  sigset(SIGINT, SIG_DFL);

  FLAGS_stderrthreshold = 0;
  vineyard::logging::InitGoogleLogging("vineyard");
  vineyard::logging::InstallFailureSignalHandler();

  // process common args
  struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
  int ret = process_args(args, argc, argv);
  if (ret != 0) {
    return ret;
  }
  // process conn args
  struct fuse_conn_info_opts* conn_opts = fuse_parse_conn_info_opts(&args);
  vineyard::fuse::fs::state.conn_opts = conn_opts;
  LOG(INFO) << "Starting vineyard fuse driver ...";
  ret = fuse_main(args.argc, args.argv, &vineyard_fuse_operations, NULL);
  fuse_opt_free_args(&args);
  return ret;
}
