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

#include "io/io/io_factory.h"

#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "glog/logging.h"
#include "network/uri.hpp"
#include "network/uri/uri_io.hpp"

#include "io/io/kafka_io_adaptor.h"
#include "io/io/local_io_adaptor.h"

namespace vineyard {

void IOFactory::Init() { LocalIOAdaptor::Init(); }

void IOFactory::Finalize() { LocalIOAdaptor::Finalize(); }

/** Create an I/O adaptor.
 * @param location the file location.
 *
 * @return a kind of I/O adaptor.
 */
std::unique_ptr<IIOAdaptor> IOFactory::CreateIOAdaptor(
    const std::string& location, Client*) {
  std::string scheme = "file";
  size_t pos = location.find_first_of(':');

  if (pos == std::string::npos || !pos || pos == location.length() - 1 ||
      location[pos + 1] != '/') {
    VLOG(1) << "Use default file location(local) to open: " + location;
  } else {
    scheme = location.substr(0, pos);
  }

  if (scheme == "file") {
    return std::unique_ptr<LocalIOAdaptor>(new LocalIOAdaptor(location));
#ifdef KAFKA_ENABLED
  } else if (scheme == "kafka") {
    return std::unique_ptr<KafkaIOAdaptor>(new KafkaIOAdaptor(location));
#endif
  }
  LOG(ERROR) << "Unimplemented adaptor for the scheme: " << scheme;
  return nullptr;
}

}  // namespace vineyard
