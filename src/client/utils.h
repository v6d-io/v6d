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

#ifndef SRC_CLIENT_UTILS_H_
#define SRC_CLIENT_UTILS_H_

#include <stdexcept>

#include "common/util/status.h"

namespace vineyard {

#ifndef ENSURE_CONNECTED
#define ENSURE_CONNECTED(this)                                 \
  if (!this->connected_) {                                     \
    return Status::ConnectionError("Client is not connected"); \
  }                                                            \
  std::lock_guard<std::recursive_mutex> __guard(this->client_mutex_)
#endif  // ENSURE_CONNECTED

}  // namespace vineyard

#endif  // SRC_CLIENT_UTILS_H_
