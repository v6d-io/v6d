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

#include "server/util/proc.h"

#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <libproc.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <unistd.h>
#endif

#include "boost/bind.hpp"     // IWYU pragma: keep
#include "boost/process.hpp"  // IWYU pragma: keep
#include "gulrak/filesystem.hpp"

#include "common/util/asio.h"  // IWYU pragma: keep

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
                    const std::vector<std::string>& args) {
  this->Start(command, args, boost::bind(&Process::recordLog, this, _1, _2),
              false);
}

void Process::Start(const std::string& command,
                    const std::vector<std::string>& args,
                    callback_t<const std::string&> callback, bool once) {
  // step 1: resolve command from PATH
  std::string command_path = command;
  {
    setenv("LC_ALL", "C", 1);
    ghc::filesystem::path target;
    if (this->findRelativeProgram(command, target).ok()) {
      command_path = target.string();
    } else {
      auto path = boost::process::search_path(command).string();
      if (!path.empty()) {
        command_path = path;
      }
    }
  }
  // launch proc
  auto env = boost::this_process::environment();
  std::error_code ec;
  proc_ = std::make_unique<boost::process::child>(
      command_path, boost::process::args(args),
      boost::process::std_in<stdin_pipe_, boost::process::std_out> stdout_pipe_,
      boost::process::std_err > stderr_pipe_, env, ec);
  if (ec) {
    VINEYARD_SUPPRESS(callback(Status::IOError("Failed to launch process '" +
                                               command + "': " + ec.message()),
                               ""));
    return;
  }

  this->AsyncRead(stdout_pipe_, stdout_buffer_, callback, once);
  this->AsyncRead(stderr_pipe_, stderr_buffer_,
                  boost::bind(&Process::recordLog, this, _1, _2), false);
}

void Process::AsyncWrite(std::string const& content, callback_t<> callback) {
  auto self(shared_from_this());
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
                        bool processed, const char delimiter) {
  auto self(shared_from_this());
  asio::async_read_until(
      pipe, buffer, delimiter,
      [self, &pipe, &buffer, callback, delimiter, once, processed](
          const boost::system::error_code& ec, std::size_t size) {
        Status status;
        std::string line;
        if (!ec || ec == asio::error::eof) {
          status = Status::OK();
          line.reserve(size + 1);
          std::istream is(&buffer);
          std::getline(is, line, delimiter);
        } else {
          status = Status::IOError("Failed to read from pipe: " + ec.message());
        }
        if (!once || !processed) {
          VINEYARD_SUPPRESS(callback(status, line));
        }
        // next round
        if (!ec) {
          self->AsyncRead(pipe, buffer, callback, delimiter, once, true);
        }
      });
}

void Process::Finish() {
  // drop a EOF marker to the input, n.b., `async_close` doesn't work.
  boost::system::error_code err;
  this->stdin_pipe_.close(err);
}

bool Process::Running() {
  if (proc_) {
    std::error_code err;
    return proc_->running(err);
  }
  return false;
}

void Process::Terminate() {
  if (proc_) {
    std::error_code err;
    proc_->terminate(err);
    proc_.reset(nullptr);
  }
}

void Process::Detach() {
  if (proc_ && proc_->valid()) {
    proc_->detach();
  }
}

void Process::Wait() {
  if (proc_ && proc_->valid()) {
    std::error_code err;
    proc_->wait(err);
  }
}

int Process::ExitCode() {
  if (proc_) {
    return proc_->exit_code();
  }
  return 0;
}

Status Process::recordLog(Status const& status, std::string const& line) {
  if (status.ok()) {
    diagnostic_.push_back(line);
  } else {
    diagnostic_.push_back(status.ToString());
    if (!line.empty()) {
      diagnostic_.push_back(line);
    }
  }
  return status;
}

Status Process::findRelativeProgram(std::string const& name,
                                    ghc::filesystem::path& target) {
  // try directly finding first
  if (ghc::filesystem::exists(name)) {
    target = ghc::filesystem::path(name);
    return Status::OK();
  }

  ghc::filesystem::path current_location;

#if defined(__APPLE__) && defined(__MACH__)
  char pathbuf[PROC_PIDPATHINFO_MAXSIZE];
  if (proc_pidpath(boost::this_process::get_id(), pathbuf, sizeof(pathbuf))) {
    current_location = std::string(pathbuf);
  } else {
    return Status::IOError("Failed to get location of current process");
  }

#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count > 0) {
    current_location = std::string(result, count);
  } else {
    return Status::IOError("Failed to get location of current process");
  }

#endif

  ghc::filesystem::path parent_path = current_location.parent_path();
  target = parent_path.append(name);
  std::error_code err;
  if (ghc::filesystem::exists(target, err)) {
    return Status::OK();
  } else {
    return Status::IOError("Failed to get location of current process");
  }
}

}  // namespace vineyard
