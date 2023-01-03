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

#ifndef SRC_COMMON_UTIL_VERSION_H_
#define SRC_COMMON_UTIL_VERSION_H_

#include <limits>
#include <string>

#include "common/util/config.h"
#include "common/util/macros.h"

namespace vineyard {

inline const char* vineyard_version() { return VINEYARD_VERSION_STRING; }

inline const bool parse_version(const char* version, int& major, int& minor,
                                int& patch) {
  char* end = nullptr;
  major = strtol(version, &end, 10);
  if (end == nullptr || *end == '\0') {
    return false;
  }
  minor = strtol(end + 1, &end, 10);
  if (end == nullptr || *end == '\0') {
    return false;
  }
  if (end == nullptr || *end == '\0') {
    return false;
  }
  patch = strtol(end + 1, &end, 10);
  return end != nullptr && *end == '\0';
}

/** Note [Semantic Versioning]
 *
 * MAJOR version when you make incompatible API changes,
 * MINOR version when you add functionality in a backwards compatible manner,
 * and PATCH version when you make backwards compatible bug fixes.
 *
 * See also: https://semver.org/
 */

/**
 * @brief Use by server to check if the incoming client is compatible.
 *
 * If the client is compatible, its version is less than or equal to server,
 * that means, the server can have backwards compatible changes.
 */
inline bool compatible_client(const char* version) {
  static int __major = 0, __minor = 0, __patch = 0;
  static bool __attribute__((used)) __parsed =
      parse_version(vineyard_version(), __major, __minor, __patch);

  int major = 0, minor = 0, patch = 0;
  return parse_version(version, major, minor, patch) && major == __major &&
         minor <= __minor;
}

inline bool compatible_client(const std::string& version) {
  return compatible_client(version.c_str());
}

/**
 * @brief Use by server to check if the incoming client is compatible.
 *
 * If the server is compatible, its version is greater than or equal to client,
 * that means, the server can have backwards compatible changes.
 */
inline bool compatible_server(const char* version) {
  static int __major = 0, __minor = 0, __patch = 0;
  static bool __attribute__((used)) __parsed =
      parse_version(vineyard_version(), __major, __minor, __patch);

  int major = 0, minor = 0, patch = 0;
  return parse_version(version, major, minor, patch) && major == __major &&
         minor >= __minor;
}

inline bool compatible_server(const std::string& version) {
  return compatible_server(version.c_str());
}

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_VERSION_H_
