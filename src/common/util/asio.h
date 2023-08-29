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

#ifndef SRC_COMMON_UTIL_ASIO_H_
#define SRC_COMMON_UTIL_ASIO_H_

#include "boost/asio.hpp"  // IWYU pragma: keep

#if BOOST_VERSION < 106600

namespace boost {
namespace asio {

using io_context = io_service;

}  // namespace asio
}  // namespace boost

#endif  // BOOST_VERSION

namespace vineyard {

namespace asio = boost::asio;  // NOLINT

// return the status if the boost::system:error_code is not OK.
#ifndef RETURN_ON_ASIO_ERROR
#define RETURN_ON_ASIO_ERROR(ec)                                        \
  do {                                                                  \
    auto _ret = (ec);                                                   \
    if (_ret) {                                                         \
      return Status::IOError("Error in boost asio: " + _ret.message()); \
    }                                                                   \
  } while (0)
#endif  // RETURN_ON_ASIO_ERROR

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_ASIO_H_
