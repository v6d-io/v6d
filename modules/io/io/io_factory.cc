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

#include <limits.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/uri.h"
#include "glog/logging.h"

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
  size_t arg_pos = location.find_first_of('#');
  std::string location_to_parse = location.substr(0, arg_pos);
  arrow::internal::Uri uri;
  {
    auto s = uri.Parse(location_to_parse);
    if (!s.ok()) {
      // defaulting to local file system: resolve to abs path first
      char resolved_path[PATH_MAX];
      realpath(location_to_parse.c_str(), resolved_path);
      location_to_parse = std::string(resolved_path);
      auto s = uri.Parse("file://" + location_to_parse);
      if (!s.ok()) {
        LOG(ERROR) << "Failed to detect the scheme of given location "
                   << location;
        return nullptr;
      }
    }
  }
  if (arg_pos != std::string::npos) {
    location_to_parse += location.substr(arg_pos);
  }

  if (uri.scheme() == "file" || uri.scheme() == "hdfs" ||
      uri.scheme() == "s3") {
    return std::unique_ptr<LocalIOAdaptor>(
        new LocalIOAdaptor(location_to_parse));
#ifdef KAFKA_ENABLED
  } else if (uri.scheme() == "kafka") {
    return std::unique_ptr<KafkaIOAdaptor>(
        new KafkaIOAdaptor(location_to_parse));
#endif
  }
  LOG(ERROR) << "Unimplemented adaptor for the scheme: " << uri.scheme()
             << " of location " << location;
  return nullptr;
}

}  // namespace vineyard
