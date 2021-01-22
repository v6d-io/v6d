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

#include "server/util/proc.h"

#include <memory>
#include <string>
#include <vector>

#include "boost/asio.hpp"
#include "boost/bind.hpp"
#include "boost/filesystem.hpp"
#include "boost/process.hpp"

#include "common/util/boost.h"

namespace vineyard {

Process::Process(asio::io_context& context)
    : stdin_pipe_(context), stdout_pipe_(context), stderr_pipe_(context) {}

Process::~Process() {
  this->Terminate();
  this->Wait();
  boost::system::error_code err;
  this->stdin_pipe_.close(err);
  this->stdout_pipe_.close(err);
  this->stderr_pipe_.close(err);
}

void Process::Start(const std::string& command,
                    const std::vector<std::string>& args,
                    callback_t<const std::string&> callback, bool once) {
  // step 1: resolve command from PATH
  std::string command_path = command;
  {
    setenv("LC_ALL", "C", 1);
    auto path = boost::process::search_path("etcd").string();
    if (!path.empty()) {
      command_path = path;
    }
  }
  // launch proc
  auto env = boost::this_process::environment();
  std::error_code ec;
  proc_ = std::make_unique<boost::process::child>(
      command_path, boost::process::args(args),
      boost::process::std_in<stdin_pipe_, boost::process::std_out> stdout_pipe_,
      boost::process::std_err > stderr_pipe_, env, ec);

  this->AsyncRead(stdout_pipe_, stdout_buffer_, callback, once);
  this->AsyncRead(stderr_pipe_, stderr_buffer_,
                  boost::bind(&Process::recordLog, this, _1, _2), false);
}

void Process::AsyncWrite(std::string const& content, callback_t<> callback) {
  auto self(this->shared_from_this());
  asio::async_write(
      stdin_pipe_, boost::asio::buffer(content.data(), content.size()),
      [self, callback](const boost::system::error_code& ec, std::size_t) {
        if (!ec) {
          VINEYARD_SUPPRESS(callback(Status::OK()));
        } else {
          VINEYARD_SUPPRESS(callback(
              Status::IOError("Failed to write to pipe: " + ec.message())));
        }
      });
}

void Process::AsyncRead(boost::process::async_pipe& pipe,
                        asio::streambuf& buffer,
                        callback_t<const std::string&> callback, bool once,
                        bool processed, const char delimeter) {
  auto self(this->shared_from_this());
  asio::async_read_until(
      pipe, buffer, delimeter,
      [self, &pipe, &buffer, callback, delimeter, once, processed](
          const boost::system::error_code& ec, std::size_t size) {
        Status status;
        std::string line;
        if (!ec || ec == asio::error::eof) {
          status = Status::OK();
          line.reserve(size + 1);
          std::istream is(&buffer);
          std::getline(is, line, delimeter);
        } else {
          status = Status::IOError("Failed to read from pipe: " + ec.message());
        }
        if (!once || !processed) {
          VINEYARD_SUPPRESS(callback(status, line));
        }
        // next round
        if (!ec) {
          self->AsyncRead(pipe, buffer, callback, delimeter, once, true);
        }
      });
}

Status Process::recordLog(Status const& status, std::string const& line) {
  if (status.ok()) {
    diagnostic_.push_back(line);
  } else {
    diagnostic_.push_back(status.ToString());
  }
  return status;
}

}  // namespace vineyard
