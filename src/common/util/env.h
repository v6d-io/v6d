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

The implementation of `get_rss`, `get_shared_rss` and `get_peek_rss`
are referred from https://stackoverflow.com/a/14927379/5080177, which
has the following license header originally:

  * Author:  David Robert Nadeau
  * Site:    http://NadeauSoftware.com/
  * License: Creative Commons Attribution 3.0 Unported License
  *          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#ifndef SRC_COMMON_UTIL_ENV_H_
#define SRC_COMMON_UTIL_ENV_H_

#include <sys/param.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>
#include <thread>

namespace vineyard {

inline std::string read_env(const char* name,
                            std::string const& default_value) {
  if (const char* envp = std::getenv(name)) {
    return std::string(envp);
  }
  return default_value;
}

inline std::string read_env(std::string const& name,
                            std::string const& default_value) {
  return read_env(name.c_str(), default_value);
}

inline std::string read_env(const char* name) { return read_env(name, ""); }

inline std::string read_env(std::string const& name) {
  return read_env(name, "");
}

inline std::string get_hostname() {
  auto hname = read_env("MY_HOST_NAME");
  if (!hname.empty()) {
    return hname;
  } else {
    char hostname_value[MAXHOSTNAMELEN];
    gethostname(&hostname_value[0], MAXHOSTNAMELEN);
    return std::string(hostname_value);
  }
}

inline std::string get_nodename() {
  auto hname = read_env("MY_NODE_NAME");
  if (!hname.empty()) {
    return hname;
  } else {
    return get_hostname();
  }
}

inline int get_pid() { return static_cast<int>(getpid()); }

inline std::thread::id get_tid() { return std::this_thread::get_id(); }

/**
 * @brief Returns the current resident set size (physical memory use) measured
 * in bytes.
 */
size_t get_rss();

/**
 * @brief Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes.
 */
size_t get_shared_rss();

/**
 * @brief Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes.
 */
size_t get_peek_rss();

/**
 * @brief Return the size limitation of available shared memory.
 */
int64_t get_maximum_shared_memory();

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_ENV_H_
