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

#include "server/util/kubectl.h"

#include <memory>
#include <string>
#include <vector>

#include "boost/asio.hpp"
#include "boost/bind.hpp"
#include "boost/filesystem.hpp"
#include "boost/process.hpp"

#include "common/util/boost.h"

namespace vineyard {

Kubectl::Kubectl(asio::io_context& context) : proc_(new Process(context)) {
  proc_->Start("kubectl", {"apply", "-f", "-"},
               [](Status const&, const std::string&) { return Status::OK(); });
}

Kubectl::~Kubectl() { proc_->Terminate(); }

void Kubectl::Apply(const std::string& content, callback_t<> callback) {
  proc_->AsyncWrite(content, [this, callback](Status const& status) {
    // TODO: improve the error diagnostic
    if (!status.ok()) {
      for (auto const& line : Diagnostic()) {
        VLOG(2) << "kubectl: " << line;
      }
    }
    return callback(status);
  });
}

}  // namespace vineyard
