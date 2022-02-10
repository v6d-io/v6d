/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include <dlfcn.h>

#include <limits.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/util/uri.h"
#include "boost/algorithm/string.hpp"
#include "glog/logging.h"

#include "common/util/env.h"

namespace vineyard {

void IOFactory::Init() {
  // load other io adaptors using dlopen
  auto other_io_adaptors = read_env("VINEYARD_OTHER_IO_ADAPTORS");
  std::vector<std::string> adaptors;
  ::boost::split(adaptors, other_io_adaptors,
                 ::boost::is_any_of(std::string(1, ':')));
  for (auto const& adaptor : adaptors) {
    if (!adaptor.empty()) {
      if (dlopen(adaptor.c_str(), RTLD_GLOBAL | RTLD_NOW) == nullptr) {
        LOG(WARNING) << "Failed to load io adaptors " << adaptor
                     << ", reason = " << dlerror();
      }
    }
  }
}

void IOFactory::Finalize() {}

/** Create an I/O adaptor.
 * @param location the file location.
 *
 * @return a kind of I/O adaptor.
 */
std::unique_ptr<IIOAdaptor> IOFactory::CreateIOAdaptor(
    const std::string& location, Client* client) {
  size_t arg_pos = location.find_first_of('#');
  std::string location_to_parse = location.substr(0, arg_pos);
  size_t i = 0;
  for (i = 0; i < location_to_parse.size(); ++i) {
    if (location_to_parse[i] < 0 || location_to_parse[i] > 127) {
      break;
    }
  }
  // If there are non-ascii charaters, we shall pre-encode the location,
  // as the arrow::internal::Uri will fail.
  auto encoded_location =
      location_to_parse.substr(0, i) +
      arrow::internal::UriEscape(location_to_parse.substr(i));
  arrow::internal::Uri uri;
  {
    auto s = uri.Parse(encoded_location);
    if (!s.ok()) {  // Assume it's a local file
      // defaulting to local file system: resolve to abs path first
      char resolved_path[PATH_MAX];
      char* res = realpath(location_to_parse.c_str(), resolved_path);
      if (!res) {
        VLOG(2) << "Warning: failed to resolve realpath of "
                << location_to_parse;
      }
      // Note we should not encode the leading '/' if there is one,
      // cause arrow::internal::Uri requires the path must be an absolute path.
      location_to_parse = std::string(resolved_path);
      auto s = uri.Parse(
          "file:///" + arrow::internal::UriEscape(location_to_parse.substr(1)));
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

  auto& known_ios = IOFactory::getKnownAdaptors();
  auto maybe_io = known_ios.find(uri.scheme());
  if (maybe_io != known_ios.end()) {
    return maybe_io->second(location_to_parse, client);
  } else {
    LOG(ERROR) << "Unimplemented adaptor for the scheme: " << uri.scheme()
               << " of location " << location;
    return nullptr;
  }
}

bool IOFactory::Register(std::string const& kind,
                         IOFactory::io_initializer_t initializer) {
  auto& known_ios = getKnownAdaptors();
  known_ios.emplace(kind, initializer);
  return true;
}

bool IOFactory::Register(std::vector<std::string> const& kinds,
                         IOFactory::io_initializer_t initializer) {
  auto& known_ios = getKnownAdaptors();
  for (auto const& kind : kinds) {
    known_ios.emplace(kind, initializer);
  }
  return true;
}

std::unordered_map<std::string, IOFactory::io_initializer_t>&
IOFactory::getKnownAdaptors() {
  static std::unordered_map<std::string, io_initializer_t>* known_adaptors =
      new std::unordered_map<std::string, io_initializer_t>();
  return *known_adaptors;
}

}  // namespace vineyard
