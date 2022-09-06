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
*/

#ifndef SRC_SERVER_UTIL_PROC_H_
#define SRC_SERVER_UTIL_PROC_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/process.hpp"

#include "common/util/asio.h"
#include "common/util/callback.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

class Process : public std::enable_shared_from_this<Process> {
 public:
  explicit Process(asio::io_context& context);

  ~Process();

  void Start(const std::string& command, const std::vector<std::string>& args,
             callback_t<const std::string&> callback, bool once = true);

  void AsyncWrite(std::string const& content, callback_t<> callback);

  void AsyncRead(boost::process::async_pipe& pipe, asio::streambuf& buffer,
                 callback_t<const std::string&> callback, bool once = true,
                 bool processed = false, const char delimeter = '\n');

  void Finish() {
    // drop a EOF marker to the input, n.b., `async_close` doesn't work.
    boost::system::error_code err;
    this->stdin_pipe_.close(err);
  }

  bool Running() {
    if (proc_) {
      std::error_code err;
      return proc_->running(err);
    }
    return false;
  }

  void Terminate() {
    if (proc_) {
      std::error_code err;
      proc_->terminate(err);
      proc_.reset(nullptr);
    }
  }

  void Detach() {
    if (proc_ && proc_->valid()) {
      proc_->detach();
    }
  }

  void Wait() {
    if (proc_ && proc_->valid()) {
      std::error_code err;
      proc_->wait(err);
    }
  }

  int ExitCode() {
    if (proc_) {
      return proc_->exit_code();
    }
    return 0;
  }

  std::list<std::string> const& Diagnostic() const { return diagnostic_; }

 private:
  std::unique_ptr<boost::process::child> proc_;
  std::list<std::string> diagnostic_;
  asio::streambuf stdout_buffer_, stderr_buffer_;
  boost::process::async_pipe stdin_pipe_, stdout_pipe_, stderr_pipe_;

  Status recordLog(Status const& status, std::string const& line);

  Status findRelativeProgram(std::string const& name,
                             boost::filesystem::path& target);
};

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_PROC_H_
