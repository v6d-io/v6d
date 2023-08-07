// Copyright 2020-2023 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env::VarError as EnvVarError;
use std::io::Error as IOError;
use std::num::{ParseFloatError, ParseIntError, TryFromIntError};
use std::sync::PoisonError;

use num_derive::{FromPrimitive, ToPrimitive};
use serde_json::Error as JSONError;
use thiserror::Error;

use super::uuid::ObjectID;

#[derive(Debug, Clone, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum StatusCode {
    OK = 0,
    Invalid = 1,
    KeyError = 2,
    TypeError = 3,
    IOError = 4,
    EndOfFile = 5,
    NotImplemented = 6,
    AssertionFailed = 7,
    UserInputError = 8,

    ObjectExists = 11,
    ObjectNotExists = 12,
    ObjectSealed = 13,
    ObjectNotSealed = 14,
    ObjectIsBlob = 15,
    ObjectTypeError = 16,
    ObjectSpilled = 17,
    ObjectNotSpilled = 18,

    MetaTreeInvalid = 21,
    MetaTreeTypeInvalid = 22,
    MetaTreeTypeNotExists = 23,
    MetaTreeNameInvalid = 24,
    MetaTreeNameNotExists = 25,
    MetaTreeLinkInvalid = 26,
    MetaTreeSubtreeNotExists = 27,

    VineyardServerNotReady = 31,
    ArrowError = 32,
    ConnectionFailed = 33,
    ConnectionError = 34,
    EtcdError = 35,
    AlreadyStopped = 36,
    RedisError = 37,

    NotEnoughMemory = 41,
    StreamDrained = 42,
    StreamFailed = 43,
    InvalidStreamState = 44,
    StreamOpened = 45,

    GlobalObjectInvalid = 51,

    UnknownError = 255,
}

#[derive(Error, Debug, Clone)]
pub struct VineyardError {
    pub code: StatusCode,
    pub message: String,
}

impl From<IOError> for VineyardError {
    fn from(error: IOError) -> Self {
        VineyardError {
            code: StatusCode::IOError,
            message: format!("internal io error: {}", error),
        }
    }
}

impl From<EnvVarError> for VineyardError {
    fn from(error: EnvVarError) -> Self {
        VineyardError {
            code: StatusCode::IOError,
            message: format!("env var error: {}", error),
        }
    }
}

impl From<ParseIntError> for VineyardError {
    fn from(error: ParseIntError) -> Self {
        VineyardError {
            code: StatusCode::IOError,
            message: format!("parse int error: {}", error),
        }
    }
}

impl From<ParseFloatError> for VineyardError {
    fn from(error: ParseFloatError) -> Self {
        VineyardError {
            code: StatusCode::IOError,
            message: format!("parse float error: {}", error),
        }
    }
}

impl From<TryFromIntError> for VineyardError {
    fn from(error: TryFromIntError) -> Self {
        VineyardError {
            code: StatusCode::IOError,
            message: format!("try from int error: {}", error),
        }
    }
}

impl<T> From<PoisonError<T>> for VineyardError {
    fn from(error: PoisonError<T>) -> Self {
        VineyardError {
            code: StatusCode::Invalid,
            message: format!("lock poison error: {}", error),
        }
    }
}

impl From<JSONError> for VineyardError {
    fn from(error: JSONError) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeInvalid,
            message: error.to_string(),
        }
    }
}

impl std::fmt::Display for VineyardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vineyard error {:?}: {}", self.code, self.message)
    }
}

impl std::default::Default for StatusCode {
    fn default() -> Self {
        StatusCode::OK
    }
}

pub type Result<T> = std::result::Result<T, VineyardError>;

impl VineyardError {
    pub fn new(code: StatusCode, message: String) -> Self {
        VineyardError { code, message }
    }

    pub fn wrap(self: &Self, message: String) -> Self {
        if self.ok() {
            return self.clone();
        }
        VineyardError {
            code: self.code.clone(),
            message: format!("{}: {}", self.message, message),
        }
    }

    pub fn invalid(message: String) -> Self {
        VineyardError {
            code: StatusCode::Invalid,
            message: message,
        }
    }

    pub fn key_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::KeyError,
            message: message,
        }
    }

    pub fn type_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::TypeError,
            message: message,
        }
    }

    pub fn io_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::IOError,
            message: message,
        }
    }

    pub fn end_of_file(message: String) -> Self {
        VineyardError {
            code: StatusCode::EndOfFile,
            message: message,
        }
    }

    pub fn not_implemented(message: String) -> Self {
        VineyardError {
            code: StatusCode::NotImplemented,
            message: message,
        }
    }

    pub fn assertion_failed(message: String) -> Self {
        VineyardError {
            code: StatusCode::AssertionFailed,
            message: message,
        }
    }

    pub fn user_input_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::UserInputError,
            message: message,
        }
    }

    pub fn object_exists(message: String) -> Self {
        VineyardError {
            code: StatusCode::ObjectExists,
            message: message,
        }
    }

    pub fn object_not_exists(message: String) -> Self {
        VineyardError {
            code: StatusCode::ObjectNotExists,
            message: message,
        }
    }

    pub fn object_sealed(message: String) -> Self {
        VineyardError {
            code: StatusCode::ObjectSealed,
            message: message,
        }
    }

    pub fn object_not_sealed(message: String) -> Self {
        VineyardError {
            code: StatusCode::ObjectNotSealed,
            message: message,
        }
    }

    pub fn object_is_blob(message: String) -> Self {
        VineyardError {
            code: StatusCode::ObjectIsBlob,
            message: message,
        }
    }

    pub fn object_type_error(expected: String, actual: String) -> Self {
        VineyardError {
            code: StatusCode::ObjectTypeError,
            message: format!("expect typename '{}', but got '{}'", expected, actual),
        }
    }

    pub fn object_spilled(object_id: ObjectID) -> Self {
        VineyardError {
            code: StatusCode::ObjectSpilled,
            message: format!("object '{}' has already been spilled", object_id),
        }
    }

    pub fn object_not_spilled(object_id: ObjectID) -> Self {
        VineyardError {
            code: StatusCode::ObjectNotSpilled,
            message: format!("object '{}' hasn't been spilled yet", object_id),
        }
    }

    pub fn meta_tree_invalid(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeInvalid,
            message: message,
        }
    }

    pub fn meta_tree_type_invalid(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeTypeInvalid,
            message: message,
        }
    }

    pub fn meta_tree_type_not_exists(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeTypeNotExists,
            message: message,
        }
    }

    pub fn meta_tree_name_invalid(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeNameInvalid,
            message: message,
        }
    }

    pub fn meta_tree_name_not_exists(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeNameNotExists,
            message: message,
        }
    }

    pub fn meta_tree_link_invalid(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeLinkInvalid,
            message: message,
        }
    }

    pub fn meta_tree_subtree_not_exists(message: String) -> Self {
        VineyardError {
            code: StatusCode::MetaTreeSubtreeNotExists,
            message: message,
        }
    }

    pub fn vineyard_server_not_ready(message: String) -> Self {
        VineyardError {
            code: StatusCode::VineyardServerNotReady,
            message: message,
        }
    }

    pub fn connection_failed(message: String) -> Self {
        VineyardError {
            code: StatusCode::ConnectionFailed,
            message: message,
        }
    }

    pub fn etcd_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::EtcdError,
            message: message,
        }
    }

    pub fn redis_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::RedisError,
            message: message,
        }
    }

    pub fn already_stopped(message: String) -> Self {
        VineyardError {
            code: StatusCode::AlreadyStopped,
            message: message,
        }
    }

    pub fn not_enough_memory(message: String) -> Self {
        VineyardError {
            code: StatusCode::NotEnoughMemory,
            message: message,
        }
    }

    pub fn stream_drained(message: String) -> Self {
        VineyardError {
            code: StatusCode::StreamDrained,
            message: message,
        }
    }

    pub fn stream_failed(message: String) -> Self {
        VineyardError {
            code: StatusCode::StreamFailed,
            message: message,
        }
    }

    pub fn invalid_stream_state(message: String) -> Self {
        VineyardError {
            code: StatusCode::InvalidStreamState,
            message: message,
        }
    }

    pub fn stream_opened(message: String) -> Self {
        VineyardError {
            code: StatusCode::StreamOpened,
            message: message,
        }
    }

    pub fn global_object_invalid(message: String) -> Self {
        VineyardError {
            code: StatusCode::GlobalObjectInvalid,
            message: message,
        }
    }

    pub fn unknown_error(message: String) -> Self {
        VineyardError {
            code: StatusCode::UnknownError,
            message: message,
        }
    }

    pub fn ok(self: &Self) -> bool {
        return self.code == StatusCode::OK;
    }

    pub fn is_invalid(self: &Self) -> bool {
        return self.code == StatusCode::Invalid;
    }

    pub fn is_key_error(self: &Self) -> bool {
        return self.code == StatusCode::KeyError;
    }

    pub fn is_type_error(self: &Self) -> bool {
        return self.code == StatusCode::TypeError;
    }

    pub fn is_io_error(self: &Self) -> bool {
        return self.code == StatusCode::IOError;
    }

    pub fn is_end_of_file(self: &Self) -> bool {
        return self.code == StatusCode::EndOfFile;
    }

    pub fn is_not_implemented(self: &Self) -> bool {
        return self.code == StatusCode::NotImplemented;
    }

    pub fn is_assertion_failed(self: &Self) -> bool {
        return self.code == StatusCode::AssertionFailed;
    }

    pub fn is_user_input_error(self: &Self) -> bool {
        return self.code == StatusCode::UserInputError;
    }

    pub fn is_object_exists(self: &Self) -> bool {
        return self.code == StatusCode::ObjectExists;
    }

    pub fn is_object_not_exists(self: &Self) -> bool {
        return self.code == StatusCode::ObjectNotExists;
    }

    pub fn is_object_sealed(self: &Self) -> bool {
        return self.code == StatusCode::ObjectSealed;
    }

    pub fn is_object_not_sealed(self: &Self) -> bool {
        return self.code == StatusCode::ObjectNotSealed;
    }

    pub fn is_object_is_blob(self: &Self) -> bool {
        return self.code == StatusCode::ObjectIsBlob;
    }

    pub fn is_object_type_error(self: &Self) -> bool {
        return self.code == StatusCode::ObjectTypeError;
    }

    pub fn is_object_spilled(self: &Self) -> bool {
        return self.code == StatusCode::ObjectSpilled;
    }

    pub fn is_object_not_spilled(self: &Self) -> bool {
        return self.code == StatusCode::ObjectNotSpilled;
    }

    pub fn is_meta_tree_invalid(self: &Self) -> bool {
        return self.code == StatusCode::MetaTreeInvalid
            || self.code == StatusCode::MetaTreeNameInvalid
            || self.code == StatusCode::MetaTreeTypeInvalid
            || self.code == StatusCode::MetaTreeLinkInvalid;
    }

    pub fn is_meta_tree_element_not_exists(self: &Self) -> bool {
        return self.code == StatusCode::MetaTreeNameNotExists
            || self.code == StatusCode::MetaTreeTypeNotExists
            || self.code == StatusCode::MetaTreeSubtreeNotExists;
    }

    pub fn is_vineyard_server_not_ready(self: &Self) -> bool {
        return self.code == StatusCode::VineyardServerNotReady;
    }

    pub fn is_arrow_error(self: &Self) -> bool {
        return self.code == StatusCode::ArrowError;
    }

    pub fn is_connection_failed(self: &Self) -> bool {
        return self.code == StatusCode::ConnectionFailed;
    }

    pub fn is_connection_error(self: &Self) -> bool {
        return self.code == StatusCode::ConnectionError;
    }

    pub fn is_etcd_error(self: &Self) -> bool {
        return self.code == StatusCode::EtcdError;
    }

    pub fn is_already_stopped(self: &Self) -> bool {
        return self.code == StatusCode::AlreadyStopped;
    }

    pub fn is_not_enough_memory(self: &Self) -> bool {
        return self.code == StatusCode::NotEnoughMemory;
    }

    pub fn is_stream_drained(self: &Self) -> bool {
        return self.code == StatusCode::StreamDrained;
    }

    pub fn is_stream_failed(self: &Self) -> bool {
        return self.code == StatusCode::StreamFailed;
    }

    pub fn is_invalid_stream_state(self: &Self) -> bool {
        return self.code == StatusCode::InvalidStreamState;
    }

    pub fn is_stream_opened(self: &Self) -> bool {
        return self.code == StatusCode::StreamOpened;
    }

    pub fn is_global_object_invalid(self: &Self) -> bool {
        return self.code == StatusCode::GlobalObjectInvalid;
    }

    pub fn is_unknown_error(self: &Self) -> bool {
        return self.code == StatusCode::UnknownError;
    }

    pub fn code(self: &Self) -> &StatusCode {
        return &self.code;
    }

    pub fn message(self: &Self) -> &String {
        return &self.message;
    }
}

pub fn vineyard_check_ok<T>(status: Result<T>) {
    if let Err(_) = status {
        panic!("Error occurs.")
    }
}

pub fn vineyard_assert(condition: bool, message: String) -> Result<()> {
    if !condition {
        return Err(VineyardError::assertion_failed(format!(
            "assertion failed: {}",
            message
        )));
    }
    return Ok(());
}

pub fn vineyard_assert_typename(expect: &str, actual: &str) -> Result<()> {
    if expect != actual {
        return Err(VineyardError::object_type_error(
            expect.to_string(),
            actual.to_string(),
        ));
    }
    return Ok(());
}
