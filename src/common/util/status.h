/**
 * NOLINT(legal/copyright)
 *
 * The file src/common/util/status.h adapt the design from project apache
 * arrow:
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/arrow/status.h
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

#ifndef SRC_COMMON_UTIL_STATUS_H_
#define SRC_COMMON_UTIL_STATUS_H_

#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "common/util/json.h"
#include "common/util/macros.h"
#include "common/util/uuid.h"

// raise a std::runtime_error (inherits std::exception), don't FATAL
#ifndef VINEYARD_CHECK_OK
#define VINEYARD_CHECK_OK(status)                                             \
  do {                                                                        \
    auto _ret = (status);                                                     \
    if (!_ret.ok()) {                                                         \
      std::clog << "[error] Check failed: " << _ret.ToString() << " in \""    \
                << #status << "\""                                            \
                << ", in function " << __PRETTY_FUNCTION__ << ", file "       \
                << __FILE__ << ", line " << VINEYARD_TO_STRING(__LINE__)      \
                << std::endl;                                                 \
      throw std::runtime_error("Check failed: " + _ret.ToString() +           \
                               " in \"" #status "\", in function " +          \
                               std::string(__PRETTY_FUNCTION__) + ", file " + \
                               __FILE__ + ", line " +                         \
                               VINEYARD_TO_STRING(__LINE__));                 \
    }                                                                         \
  } while (0)
#endif  // VINEYARD_CHECK_OK

// check the condition, raise and runtime error, rather than `FATAL` when false
#ifndef VINEYARD_ASSERT_NO_VERBOSE
#define VINEYARD_ASSERT_NO_VERBOSE(condition)                             \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::clog << "[error] Assertion failed in \"" #condition "\""       \
                << ", in function '" << __PRETTY_FUNCTION__ << "', file " \
                << __FILE__ << ", line " << VINEYARD_TO_STRING(__LINE__)  \
                << std::endl;                                             \
      throw std::runtime_error(                                           \
          "Assertion failed in \"" #condition "\", in function '" +       \
          std::string(__PRETTY_FUNCTION__) + "', file " + __FILE__ +      \
          ", line " + VINEYARD_TO_STRING(__LINE__));                      \
    }                                                                     \
  } while (0)
#endif  // VINEYARD_ASSERT_NO_VERBOSE

// check the condition, raise and runtime error, rather than `FATAL` when false
#ifndef VINEYARD_ASSERT_VERBOSE
#define VINEYARD_ASSERT_VERBOSE(condition, message)                           \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::clog << "[error] Assertion failed in \"" #condition "\": "         \
                << std::string(message) << ", in function '"                  \
                << __PRETTY_FUNCTION__ << "', file " << __FILE__ << ", line " \
                << VINEYARD_TO_STRING(__LINE__) << std::endl;                 \
      throw std::runtime_error(                                               \
          "Assertion failed in \"" #condition "\": " + std::string(message) + \
          ", in function '" + std::string(__PRETTY_FUNCTION__) + "', file " + \
          __FILE__ + ", line " + VINEYARD_TO_STRING(__LINE__));               \
    }                                                                         \
  } while (0)
#endif  // VINEYARD_ASSERT_VERBOSE

#ifndef VINEYARD_ASSERT
#define VINEYARD_ASSERT(...)                                                  \
  GET_MACRO(__VA_ARGS__, VINEYARD_ASSERT_VERBOSE, VINEYARD_ASSERT_NO_VERBOSE) \
  (__VA_ARGS__)
#endif  // VINEYARD_ASSERT

// return the status if the status is not OK.
#ifndef RETURN_ON_ERROR
#define RETURN_ON_ERROR(status) \
  do {                          \
    auto _ret = (status);       \
    if (!_ret.ok()) {           \
      return _ret;              \
    }                           \
  } while (0)
#endif  // RETURN_ON_ERROR

// return a null pointer if the status is not OK.
#ifndef RETURN_NULL_ON_ERROR
#define RETURN_NULL_ON_ERROR(status)                                       \
  do {                                                                     \
    auto _ret = (status);                                                  \
    if (!_ret.ok()) {                                                      \
      std::clog << "[error] Check failed: " << _ret.ToString() << " in \"" \
                << #status << "\"" << std::endl;                           \
      return nullptr;                                                      \
    }                                                                      \
  } while (0)
#endif  // RETURN_NULL_ON_ERROR

// return the status if the status is not OK.
#ifndef RETURN_ON_ASSERT_NO_VERBOSE
#define RETURN_ON_ASSERT_NO_VERBOSE(condition)                \
  do {                                                        \
    if (!(condition)) {                                       \
      return ::vineyard::Status::AssertionFailed(#condition); \
    }                                                         \
  } while (0)
#endif  // RETURN_ON_ASSERT_NO_VERBOSE

// return the status if the status is not OK.
#ifndef RETURN_ON_ASSERT_VERBOSE
#define RETURN_ON_ASSERT_VERBOSE(condition, message) \
  do {                                               \
    if (!(condition)) {                              \
      return ::vineyard::Status::AssertionFailed(    \
          std::string(#condition ": ") + message);   \
    }                                                \
  } while (0)
#endif  // RETURN_ON_ASSERT_VERBOSE

#ifndef RETURN_ON_ASSERT
#define RETURN_ON_ASSERT(...)                      \
  GET_MACRO(__VA_ARGS__, RETURN_ON_ASSERT_VERBOSE, \
            RETURN_ON_ASSERT_NO_VERBOSE)           \
  (__VA_ARGS__)
#endif  // RETURN_ON_ASSERT

// return a null pointer if the status is not OK.
#ifndef RETURN_NULL_ON_ASSERT_NO_VERBOSE
#define RETURN_NULL_ON_ASSERT_NO_VERBOSE(condition)                       \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::clog << "[error] Assertion failed in \"" #condition "\""       \
                << ", in function '" << __PRETTY_FUNCTION__ << "', file " \
                << __FILE__ << ", line " << VINEYARD_TO_STRING(__LINE__)  \
                << std::endl;                                             \
    }                                                                     \
  } while (0)
#endif  // RETURN_NULL_ON_ASSERT_NO_VERBOSE

// return a null pointer if the status is not OK.
#ifndef RETURN_NULL_ON_ASSERT_VERBOSE
#define RETURN_NULL_ON_ASSERT_VERBOSE(condition, message)                     \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::clog << "[error] Assertion failed in \"" #condition "\": "         \
                << std::string(message) << ", in function '"                  \
                << __PRETTY_FUNCTION__ << "', file " << __FILE__ << ", line " \
                << VINEYARD_TO_STRING(__LINE__) << std::endl;                 \
    }                                                                         \
  } while (0)
#endif  // RETURN_NULL_ON_ASSERT_VERBOSE

// return a null pointer if the status is not OK.
#ifndef RETURN_NULL_ON_ASSERT
#define RETURN_NULL_ON_ASSERT(...)                      \
  GET_MACRO(__VA_ARGS__, RETURN_NULL_ON_ASSERT_VERBOSE, \
            RETURN_NULL_ON_ASSERT_NO_VERBOSE)           \
  (__VA_ARGS__)
#endif  // RETURN_NULL_ON_ASSERT

// discard and ignore the error status.
#ifndef VINEYARD_DISCARD
#define VINEYARD_DISCARD(status)                              \
  do {                                                        \
    auto _ret = (status);                                     \
    if (!_ret.ok()) {} /* NOLINT(whitespace/empty_if_body) */ \
  } while (0)
#endif  // VINEYARD_DISCARD

// suppress and ignore the error status, deprecated in favour of
// VINEYARD_DISCARD
#ifndef VINEYARD_SUPPRESS
#define VINEYARD_SUPPRESS(status)                             \
  do {                                                        \
    auto _ret = (status);                                     \
    if (!_ret.ok()) {} /* NOLINT(whitespace/empty_if_body) */ \
  } while (0)
#endif  // VINEYARD_SUPPRESS

// print the error message when failed, but never throw or abort.
#ifndef VINEYARD_LOG_ERROR
#define VINEYARD_LOG_ERROR(status)                                         \
  do {                                                                     \
    auto _ret = (status);                                                  \
    if (!_ret.ok()) {                                                      \
      std::clog << "[error] Check failed: " << _ret.ToString() << " in \"" \
                << #status << "\"" << std::endl;                           \
    }                                                                      \
  } while (0)
#endif  // VINEYARD_LOG_ERROR

namespace vineyard {

enum class StatusCode : unsigned char {
  kOK = 0,
  kInvalid = 1,
  kKeyError = 2,
  kTypeError = 3,
  kIOError = 4,
  kEndOfFile = 5,
  kNotImplemented = 6,
  kAssertionFailed = 7,
  kUserInputError = 8,

  kObjectExists = 11,
  kObjectNotExists = 12,
  kObjectSealed = 13,
  kObjectNotSealed = 14,
  kObjectIsBlob = 15,
  kObjectTypeError = 16,
  kObjectSpilled = 17,
  kObjectNotSpilled = 18,

  kMetaTreeInvalid = 21,
  kMetaTreeTypeInvalid = 22,
  kMetaTreeTypeNotExists = 23,
  kMetaTreeNameInvalid = 24,
  kMetaTreeNameNotExists = 25,
  kMetaTreeLinkInvalid = 26,
  kMetaTreeSubtreeNotExists = 27,

  kVineyardServerNotReady = 31,
  kArrowError = 32,
  kConnectionFailed = 33,
  kConnectionError = 34,
  kEtcdError = 35,
  kAlreadyStopped = 36,
  kRedisError = 37,

  kNotEnoughMemory = 41,
  kStreamDrained = 42,
  kStreamFailed = 43,
  kInvalidStreamState = 44,
  kStreamOpened = 45,

  kGlobalObjectInvalid = 51,

  kUnknownError = 255
};

/**
 * @brief A Status encapsulates the result of an operation.  It may indicate
 * success, or it may indicate an error with an associated error message.
 *
 * Vineyard also provides macros for convenient error handling:
 *
 * - `VINEYARD_CHECK_OK`: used for check vineyard's status, when error occurs,
 *   raise a runtime exception.
 *
 *   It should be used where the function itself doesn't return a `Status`, but
 *   we want to check the status of certain statements.
 *
 * - `VINEYARD_ASSERT`: used for assert on a condition, if false, raise a
 * runtime exception.
 *
 *   It should be used where the function itself doesn't return a `Status`, but
 *   we need to check on a condition.
 *
 * - `RETURN_ON_ERROR`: it looks like VINEYARD_CHECK_OK, but return the Status
 * if it is not ok, without causing an exception.
 *
 * - `RETURN_ON_ASSERT`: it looks like VINEYARD_ASSERT, but return a status of
 *   AssertionFailed, without causing an exception and abort the program.
 *
 * - `VINEYARD_DISCARD`: suppress and ignore the error status, just logging it
 *   out.
 *
 * - `VINEYARD_SUPPRESS`: suppress and ignore the error status, just logging it
 *   out, deprecated in favour of VINEYARD_DISCARD.
 *
 *   This one is usaully used in dtor that we shouldn't raise exceptions.
 */
class VINEYARD_MUST_USE_TYPE Status {
 public:
  Status() noexcept : state_(nullptr) {}
  ~Status() noexcept {
    if (state_ != nullptr) {
      DeleteState();
    }
  }
  Status(StatusCode, const std::string& msg);
  // Copy the specified status.
  inline Status(const Status& s);
  inline Status& operator=(const Status& s);

  // Move the specified status.
  inline Status(Status&& s) noexcept;
  inline Status& operator=(Status&& s) noexcept;

  // AND the statuses.
  inline Status operator&(const Status& s) const noexcept;
  inline Status operator&(Status&& s) const noexcept;
  inline Status& operator&=(const Status& s) noexcept;
  inline Status& operator&=(Status&& s) noexcept;
  inline Status& operator+=(const Status& s) noexcept;

  /// Return a success status
  inline static Status OK() { return Status(); }

  /// Wrap a status with customized extra message
  inline static Status Wrap(const Status& s, const std::string& message) {
    if (s.ok()) {
      return s;
    }
    return Status(s.code(), message + ": " + s.message());
  }

  /// Return an error status for invalid data (for example a string that
  /// fails parsing).
  static Status Invalid() { return Status(StatusCode::kInvalid, ""); }
  /// Return an error status for invalid data, with user specified error
  /// message.
  static Status Invalid(std::string const& message) {
    return Status(StatusCode::kInvalid, message);
  }

  /// Return an error status for failed key lookups (e.g. column name in a
  /// table).
  static Status KeyError() { return Status(StatusCode::kKeyError, ""); }
  static Status KeyError(std::string const& msg) {
    return Status(StatusCode::kKeyError, msg);
  }

  /// Return an error status for type errors (such as mismatching data types).
  static Status TypeError() { return Status(StatusCode::kTypeError, ""); }

  /// Return an error status for IO errors (e.g. Failed to open or read from a
  /// file).
  static Status IOError(const std::string& msg = "") {
    return Status(StatusCode::kIOError, msg);
  }

  /// Return an error status when reader reaches at the end of file.
  static Status EndOfFile() { return Status(StatusCode::kEndOfFile, ""); }

  /// Return an error status when an operation or a combination of operation and
  /// data types is unimplemented
  static Status NotImplemented(std::string const& message = "") {
    return Status(StatusCode::kNotImplemented, message);
  }

  /// Return an error status when the condition assertion is false.
  static Status AssertionFailed(std::string const& condition) {
    return Status(StatusCode::kAssertionFailed, condition);
  }

  /// Return a status code indicates invalid user input.
  static Status UserInputError(std::string const& message = "") {
    return Status(StatusCode::kUserInputError, message);
  }

  /// Return an error when the object exists if the user are still trying to
  /// creating such object.
  static Status ObjectExists() { return Status(StatusCode::kObjectExists, ""); }

  /// Return an error when the object exists if the user are still trying to
  /// creating such object.
  static Status ObjectExists(std::string const& message) {
    return Status(StatusCode::kObjectExists, message);
  }

  /// Return an error when user want to get an object however the target object
  /// not exists.
  static Status ObjectNotExists(std::string const& message = "") {
    return Status(StatusCode::kObjectNotExists, message);
  }

  /// Return an error when user are trying to seal a builder but the builder has
  /// already been sealed.
  static Status ObjectSealed(std::string const& message = "") {
    return Status(StatusCode::kObjectSealed, message);
  }

  /// Return an error when user are trying to maniplate the object but the
  /// object hasn't been sealed yet.
  static Status ObjectNotSealed() {
    return Status(StatusCode::kObjectNotSealed, "");
  }

  /// Return an error when user are trying to maniplate the object but the
  /// object hasn't been sealed yet.
  static Status ObjectNotSealed(std::string const& message) {
    return Status(StatusCode::kObjectNotSealed, message);
  }

  /// Return an error when user are trying to perform unsupported operations
  /// on blob objects.
  static Status ObjectIsBlob(std::string const& message = "") {
    return Status(StatusCode::kObjectIsBlob, message);
  }

  /// Return an error when user are trying to perform cast object to mismatched
  /// types.
  static Status ObjectTypeError(std::string const& expect,
                                std::string const& actual) {
    return Status(StatusCode::kObjectTypeError,
                  "expect '" + expect + "', but got '" + actual + "'");
  }

  /// Return an error when the object has already been spilled.
  static Status ObjectSpilled(const ObjectID& object_id) {
    return Status(StatusCode::kObjectSpilled, "object '" +
                                                  ObjectIDToString(object_id) +
                                                  "' has already been spilled");
  }

  /// Return an error when the object is not spilled yet.
  static Status ObjectNotSpilled(const ObjectID& object_id) {
    return Status(
        StatusCode::kObjectNotSpilled,
        "object '" + ObjectIDToString(object_id) + "' hasn't been spilled yet");
  }

  /// Return an error when metatree related error occurs.
  static Status MetaTreeInvalid(std::string const& message = "") {
    return Status(StatusCode::kMetaTreeInvalid, message);
  }

  /// Return an error if the "typename" field in metatree is invalid.
  static Status MetaTreeTypeInvalid() {
    return Status(StatusCode::kMetaTreeTypeInvalid, "");
  }

  /// Return an error if the "typename" field in metatree is invalid.
  static Status MetaTreeTypeInvalid(std::string const& message) {
    return Status(StatusCode::kMetaTreeTypeInvalid, message);
  }

  /// Return an error if the "typename" field not exists in metatree.
  static Status MetaTreeTypeNotExists() {
    return Status(StatusCode::kMetaTreeTypeNotExists, "");
  }

  /// Return an error if the "typename" field not exists in metatree.
  static Status MetaTreeTypeNotExists(std::string const& message) {
    return Status(StatusCode::kMetaTreeTypeNotExists, message);
  }

  /// Return an error if the "id" field in metatree is invalid.
  static Status MetaTreeNameInvalid() {
    return Status(StatusCode::kMetaTreeNameInvalid, "");
  }

  /// Return an error if the "id" field in metatree is invalid.
  static Status MetaTreeNameInvalid(std::string const& message) {
    return Status(StatusCode::kMetaTreeNameInvalid, message);
  }

  /// Return an error if the "id" field not exists in metatree.
  static Status MetaTreeNameNotExists() {
    return Status(StatusCode::kMetaTreeNameNotExists, "");
  }

  /// Return an error if the "id" field not exists in metatree.
  static Status MetaTreeNameNotExists(std::string const& message) {
    return Status(StatusCode::kMetaTreeNameNotExists, message);
  }

  /// Return an error if a field in metatree is expected to be a _link_ but it
  /// isn't.
  static Status MetaTreeLinkInvalid() {
    return Status(StatusCode::kMetaTreeLinkInvalid, "");
  }

  /// Return an error if a field in metatree is expected to be a _link_ but it
  /// isn't.
  static Status MetaTreeLinkInvalid(std::string const& message) {
    return Status(StatusCode::kMetaTreeLinkInvalid, message);
  }

  /// Return an error when expected subtree doesn't exist in metatree.
  static Status MetaTreeSubtreeNotExists() {
    return Status(StatusCode::kMetaTreeSubtreeNotExists, "");
  }

  /// Return an error when expected subtree doesn't exist in metatree.
  static Status MetaTreeSubtreeNotExists(std::string const& key) {
    return Status(StatusCode::kMetaTreeSubtreeNotExists, key);
  }

  /// Return an error when the requested vineyard server is not ready yet.
  static Status VineyardServerNotReady(std::string const& message) {
    return Status(StatusCode::kVineyardServerNotReady, message);
  }

  /// Return an error when client failed to connect to vineyard server.
  static Status ConnectionFailed(std::string const& message = "") {
    return Status(StatusCode::kConnectionFailed,
                  "Failed to connect to vineyardd: " + message);
  }

  /// Return an error when client losts connection to vineyard server.
  static Status ConnectionError(std::string const& message = "") {
    return Status(StatusCode::kConnectionError, message);
  }

  /// Return an error when the vineyard server meets an etcd related error.
  static Status EtcdError(std::string const& error_message) {
    return Status(StatusCode::kEtcdError, error_message);
  }

  /// Return an error when the vineyard server meets an etcd related error, with
  /// etcd error code embedded.
  static Status EtcdError(int error_code, std::string const& error_message) {
    if (error_code == 0) {
      return Status::OK();
    }
    return Status(StatusCode::kEtcdError, error_message + ", error code: " +
                                              std::to_string(error_code));
  }

  /// Return an error when the vineyard server meets an redis related error.
  static Status RedisError(std::string const& error_message) {
    return Status(StatusCode::kRedisError, error_message);
  }

  /// Return an error when the vineyard server meets an redis related error,
  /// with redis error code embedded.
  static Status RedisError(int error_code, std::string const& error_message) {
    if (error_code == 0) {
      return Status::OK();
    }
    return Status(StatusCode::kRedisError, error_message + ", error code: " +
                                               std::to_string(error_code));
  }

  /// Return an error when the vineyard server meets an redis related error,
  /// with redis error code embedded, with redis error type embedded.
  static Status RedisError(int error_code, std::string const& error_message,
                           std::string const& error_type) {
    if (error_code == 0) {
      return Status::OK();
    }
    return Status(StatusCode::kRedisError,
                  error_message + error_type +
                      ", error code: " + std::to_string(error_code));
  }

  static Status AlreadyStopped(std::string const& component = "") {
    return Status(StatusCode::kAlreadyStopped, component + " already stopped");
  }

  /// Return an error when the vineyard server cannot allocate more memory
  /// blocks.
  static Status NotEnoughMemory(std::string const& error_message) {
    return Status(StatusCode::kNotEnoughMemory, error_message);
  }

  /// Return a status code that indicates there's no more chunks in the stream.
  static Status StreamDrained() {
    return Status(StatusCode::kStreamDrained, "Stream drained: no more chunks");
  }

  /// Return a status code that indicates the stream has failed.
  static Status StreamFailed() {
    return Status(StatusCode::kStreamFailed, "Stream source failed");
  }

  /// Return a status code about internal invalid state found for stream, it
  /// usually means bugs in vineyard and please file an issue to us.
  static Status InvalidStreamState(std::string const& error_message) {
    return Status(StatusCode::kInvalidStreamState, error_message);
  }

  /// Return a status code that indicates the stream has been opened.
  static Status StreamOpened() {
    return Status(StatusCode::kStreamOpened, "Stream already opened");
  }

  /// Return a status code indicates invalid global object structure.
  static Status GlobalObjectInvalid(
      std::string const& message = "Global object cannot be nested") {
    return Status(StatusCode::kGlobalObjectInvalid, message);
  }

  /// Return an error status for unknown errors
  static Status UnknownError(std::string const& message = "") {
    return Status(StatusCode::kUnknownError, message);
  }

  /// Return true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }

  /// Return true iff the status indicates invalid data.
  bool IsInvalid() const { return code() == StatusCode::kInvalid; }
  /// Return true iff the status indicates a key lookup error.
  bool IsKeyError() const { return code() == StatusCode::kKeyError; }
  /// Return true iff the status indicates a container reaching capacity limits.
  bool IsTypeError() const { return code() == StatusCode::kTypeError; }
  /// Return true iff the status indicates an IO-related failure.
  bool IsIOError() const { return code() == StatusCode::kIOError; }
  /// Return true iff the status indicates the end of file.
  bool IsEndOfFile() const { return code() == StatusCode::kEndOfFile; }
  /// Return true iff the status indicates an unimplemented operation.
  bool IsNotImplemented() const {
    return code() == StatusCode::kNotImplemented;
  }
  /// Return true iff the status indicates an unimplemented operation.
  bool IsAssertionFailed() const {
    return code() == StatusCode::kAssertionFailed;
  }
  /// Return true iff there's some problems in user's input.
  bool IsUserInputError() const {
    return code() == StatusCode::kUserInputError;
  }
  /// Return true iff the status indicates already existing object.
  bool IsObjectExists() const { return code() == StatusCode::kObjectExists; }
  /// Return true iff the status indicates non-existing object.
  bool IsObjectNotExists() const {
    return code() == StatusCode::kObjectNotExists;
  }
  /// Return true iff the status indicates already existing object.
  bool IsObjectSealed() const { return code() == StatusCode::kObjectSealed; }
  /// Return true iff the status indicates non-sealed object.
  bool IsObjectNotSealed() const {
    return code() == StatusCode::kObjectNotSealed;
  }
  /// Return true iff the status indicates unsupported operations on blobs.
  bool IsObjectIsBlob() const { return code() == StatusCode::kObjectIsBlob; }
  /// Return true iff the status indicates object type mismatch.
  bool IsObjectTypeError() const {
    return code() == StatusCode::kObjectTypeError;
  }
  /// Return true iff the status indicates object has already been spilled.
  bool IsObjectSpilled() const { return code() == StatusCode::kObjectSpilled; }
  /// Return true iff the status indicates object is not spilled yet.
  bool IsObjectNotSpilled() const {
    return code() == StatusCode::kObjectNotSpilled;
  }
  /// Return true iff the status indicates subtree not found in metatree.
  bool IsMetaTreeSubtreeNotExists() const {
    return code() == StatusCode::kMetaTreeSubtreeNotExists;
  }
  /// Return true iff the metadata tree contains unexpected invalid meta
  bool IsMetaTreeInvalid() const {
    return code() == StatusCode::kMetaTreeInvalid ||
           code() == StatusCode::kMetaTreeNameInvalid ||
           code() == StatusCode::kMetaTreeTypeInvalid ||
           code() == StatusCode::kMetaTreeLinkInvalid;
  }
  /// Return true iff expected fields not found in metatree.
  bool IsMetaTreeElementNotExists() const {
    return code() == StatusCode::kMetaTreeNameNotExists ||
           code() == StatusCode::kMetaTreeTypeNotExists ||
           code() == StatusCode::kMetaTreeSubtreeNotExists;
  }
  /// Return true iff the vineyard server is still not ready.
  bool IsVineyardServerNotReady() const {
    return code() == StatusCode::kVineyardServerNotReady;
  }
  /// Return true iff the error is an arrow's error.
  bool IsArrowError() const { return code() == StatusCode::kArrowError; }
  /// Return true iff the client fails to setup a connection with vineyard
  /// server.
  bool IsConnectionFailed() const {
    return code() == StatusCode::kConnectionFailed;
  }
  /// Return true iff the client meets an connection error.
  bool IsConnectionError() const {
    return code() == StatusCode::kConnectionError;
  }
  /// Return true iff etcd related error occurs in vineyard server.
  bool IsEtcdError() const { return code() == StatusCode::kEtcdError; }
  /// Return true iff certain component is already stopped.
  bool IsAlreadyStopped() const {
    return code() == StatusCode::kAlreadyStopped;
  }
  /// Return true iff vineyard server fails to allocate memory.
  bool IsNotEnoughMemory() const {
    return code() == StatusCode::kNotEnoughMemory;
  }
  /// Return true iff the stream has been marked as drained.
  bool IsStreamDrained() const { return code() == StatusCode::kStreamDrained; }
  /// Return true iff the stream has been marked as failed.
  bool IsStreamFailed() const { return code() == StatusCode::kStreamFailed; }
  /// Return true iff the internal state of the stream is invalid.
  bool IsInvalidStreamState() const {
    return code() == StatusCode::kInvalidStreamState;
  }
  /// Return true iff the stream has been opened.
  bool IsStreamOpened() const { return code() == StatusCode::kStreamOpened; }
  /// Return true iff the given global object is invalid.
  bool IsGlobalObjectInvalid() const {
    return code() == StatusCode::kGlobalObjectInvalid;
  }
  /// Return true iff the status indicates an unknown error.
  bool IsUnknownError() const { return code() == StatusCode::kUnknownError; }

  /// \brief Return a string representation of this status suitable for
  /// printing.
  ///
  /// The string "OK" is returned for success.
  std::string ToString() const;

  /// \brief Return a JSON representation of this status suitable for
  /// printing.
  json ToJSON() const;

  /// \brief Return a string representation of the status code, without the
  /// message text or POSIX code information.
  std::string CodeAsString() const;

  /// \brief Return the StatusCode value attached to this status.
  StatusCode code() const { return ok() ? StatusCode::kOK : state_->code; }

  /// \brief Return the specific error message attached to this status.
  std::string message() const { return ok() ? "" : state_->msg; }

  [[noreturn]] void Abort() const;
  [[noreturn]] void Abort(const std::string& message) const;

  template <typename T>
  Status& operator<<(const T& s);
  const std::string Backtrace() const { return backtrace_; }

 private:
  struct State {
    StatusCode code;
    std::string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  State* state_;
  // OK status has a `NULL` backtrace_.
  std::string backtrace_;

  void DeleteState() {
    delete state_;
    state_ = nullptr;
  }
  void CopyFrom(const Status& s);
  void MoveFrom(Status& s);
  void MergeFrom(const Status& s);
};

inline std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const StatusCode& x) {
  os << (unsigned char) x;
  return os;
}

inline std::istream& operator>>(std::istream& is, StatusCode& x) {
  unsigned char c;
  is >> c;
  x = static_cast<StatusCode>(c);
  return is;
}

template <typename T>
Status& Status::operator<<(const T& s) {
  // CHECK_NE(state_, nullptr);  // Not allowed to append message to a OK
  // message;
  std::ostringstream tmp;
  tmp << s;
  state_->msg.append(tmp.str());
  return *this;
}

Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

Status& Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    CopyFrom(s);
  }
  return *this;
}

Status::Status(Status&& s) noexcept : state_(s.state_) { s.state_ = nullptr; }

Status& Status::operator=(Status&& s) noexcept {
  MoveFrom(s);
  return *this;
}

/// \cond FALSE
// (note: emits warnings on Doxygen < 1.8.15,
//  see https://github.com/doxygen/doxygen/issues/6295)
Status Status::operator&(const Status& s) const noexcept {
  if (ok()) {
    return s;
  } else {
    return *this;
  }
}

Status Status::operator&(Status&& s) const noexcept {
  if (ok()) {
    return std::move(s);
  } else {
    return *this;
  }
}

Status& Status::operator&=(const Status& s) noexcept {
  if (ok() && !s.ok()) {
    CopyFrom(s);
  }
  return *this;
}

Status& Status::operator&=(Status&& s) noexcept {
  if (ok() && !s.ok()) {
    MoveFrom(s);
  }
  return *this;
}

Status& Status::operator+=(const Status& s) noexcept {
  if (!s.ok()) {
    MergeFrom(s);
  }
  return *this;
}

/// \endcond

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_STATUS_H_
