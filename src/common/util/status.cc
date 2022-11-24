/**
 * NOLINT(legal/copyright)
 *
 * The file src/common/util/status.cc adapt the design from project apache
 * arrow:
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/arrow/status.cc
 *
 * which is original referred from leveldb and has the following license:
 *
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A Status encapsulates the result of an operation.  It may indicate success,
// or it may indicate an error with an associated error message.
//
// Multiple threads can invoke const methods on a Status without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Status must use
// external synchronization.
 */

#include "common/util/status.h"

#include <iostream>
#include <string>

#include "common/backtrace/backtrace.hpp"
#include "common/util/env.h"

namespace vineyard {

Status::Status(StatusCode code, const std::string& msg) {
  state_ = new State;
  state_->code = code;
  state_->msg = msg;
#ifndef NDEBUG
  static std::string vineyard_error_with_traceback =
      read_env("VINEYARD_ERROR_WITH_TRACEBACK");
  if ((!vineyard_error_with_traceback.empty()) && code != StatusCode::kOK) {
    std::stringstream ss;
    vineyard::backtrace_info::backtrace(ss, true);
    backtrace_ = ss.str();
  }
#endif
}

void Status::CopyFrom(const Status& s) {
  delete state_;
  if (s.state_ == nullptr) {
    state_ = nullptr;
  } else {
    state_ = new State(*s.state_);
  }
}

void Status::MoveFrom(Status& s) {
  delete state_;
  state_ = s.state_;
  s.state_ = nullptr;
}

void Status::MergeFrom(const Status& s) {
  delete state_;
  if (state_ == nullptr) {
    if (s.state_ != nullptr) {
      state_ = new State(*s.state_);
    }
  } else {
    if (s.state_ != nullptr) {
      state_->msg += "; " + s.state_->msg;
    }
  }
}

std::string Status::CodeAsString() const {
  if (state_ == nullptr) {
    return "OK";
  }

  const char* type;
  switch (code()) {
  case StatusCode::kOK:
    type = "OK";
    break;
  case StatusCode::kInvalid:
    type = "Invalid";
    break;
  case StatusCode::kKeyError:
    type = "Key error";
    break;
  case StatusCode::kTypeError:
    type = "Type error";
    break;
  case StatusCode::kIOError:
    type = "IOError";
    break;
  case StatusCode::kEndOfFile:
    type = "End Of File";
    break;
  case StatusCode::kNotImplemented:
    type = "Not implemented";
    break;
  case StatusCode::kAssertionFailed:
    type = "Assertion failed";
    break;
  case StatusCode::kUserInputError:
    type = "User input error";
    break;
  case StatusCode::kObjectExists:
    type = "Object exists";
    break;
  case StatusCode::kObjectNotExists:
    type = "Object not exists";
    break;
  case StatusCode::kObjectSealed:
    type = "Object sealed";
    break;
  case StatusCode::kObjectNotSealed:
    type = "Object not sealed";
    break;
  case StatusCode::kObjectIsBlob:
    type = "Object not blob";
    break;
  case StatusCode::kObjectTypeError:
    type = "Object type mismatch";
    break;
  case StatusCode::kMetaTreeInvalid:
    type = "Metatree invalid";
    break;
  case StatusCode::kMetaTreeTypeInvalid:
    type = "Metatree type invalid";
    break;
  case StatusCode::kMetaTreeTypeNotExists:
    type = "Metatree type not exists";
    break;
  case StatusCode::kMetaTreeNameInvalid:
    type = "Metatree name invalid";
    break;
  case StatusCode::kMetaTreeNameNotExists:
    type = "Metatree name not exists";
    break;
  case StatusCode::kMetaTreeLinkInvalid:
    type = "Metatree link invalid";
    break;
  case StatusCode::kMetaTreeSubtreeNotExists:
    type = "Metatree subtree not exists.";
    break;
  case StatusCode::kVineyardServerNotReady:
    type = "Vineyard server not ready";
    break;
  case StatusCode::kArrowError:
    type = "Arrow error";
    break;
  case StatusCode::kConnectionFailed:
    type = "Connection failed";
    break;
  case StatusCode::kConnectionError:
    type = "Connection error";
    break;
  case StatusCode::kEtcdError:
    type = "Etcd error";
    break;
  case StatusCode::kRedisError:
    type = "Redis error";
    break;
  case StatusCode::kNotEnoughMemory:
    type = "Not enough memory";
    break;
  case StatusCode::kStreamDrained:
    type = "Stream drain";
    break;
  case StatusCode::kStreamFailed:
    type = "Stream failed";
    break;
  case StatusCode::kInvalidStreamState:
    type = "Invalid stream state";
    break;
  case StatusCode::kStreamOpened:
    type = "Stream opened";
    break;
  case StatusCode::kGlobalObjectInvalid:
    type = "Global object invalid";
    break;
  case StatusCode::kUnknownError:
  default:
    type = "Unknown error";
    break;
  }
  return std::string(type);
}

std::string Status::ToString() const {
  std::string result(CodeAsString());
  if (state_ == nullptr) {
    return result;
  }
  result += ": ";
  result += state_->msg;
  return result;
}

json Status::ToJSON() const {
  json tree;
  tree["code"] = static_cast<int>(code());
  if (!ok() /* state_ != nullptr */) {
    tree["message"] = state_->msg;
  }
  return tree;
}

void Status::Abort() const { Abort(std::string()); }

void Status::Abort(const std::string& message) const {
  std::cerr << "-- Vineyard Fatal Error --\n";
  if (!message.empty()) {
    std::cerr << message << "\n";
  }
  std::cerr << ToString() << std::endl;
  std::abort();
}

}  // namespace vineyard
