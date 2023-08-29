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

use std::backtrace::{Backtrace, BacktraceStatus};
use std::env::VarError as EnvVarError;
use std::io::Error as IOError;
use std::num::{ParseFloatError, ParseIntError, TryFromIntError};
use std::sync::PoisonError;

use num_derive::{FromPrimitive, ToPrimitive};
use serde_json::Error as JSONError;

use super::uuid::ObjectID;

#[derive(Debug, Clone, Default, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum StatusCode {
    #[default]
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

pub struct VineyardError {
    pub code: StatusCode,
    pub message: String,
    pub backtrace: Backtrace,
}

impl std::error::Error for VineyardError {}

impl std::fmt::Debug for VineyardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.backtrace.status() == BacktraceStatus::Captured {
            write!(
                f,
                "{:?}: {}\nBacktrace: {:?}",
                self.code, self.message, self.backtrace
            )
        } else {
            write!(f, "{:?}: {}", self.code, self.message)
        }
    }
}

impl std::fmt::Display for VineyardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vineyard error {:?}: {}", self.code, self.message)
    }
}

impl From<IOError> for VineyardError {
    fn from(error: IOError) -> Self {
        VineyardError::new(StatusCode::IOError, format!("internal io error: {}", error))
    }
}

impl From<EnvVarError> for VineyardError {
    fn from(error: EnvVarError) -> Self {
        VineyardError::new(StatusCode::IOError, format!("env var error: {}", error))
    }
}

impl From<ParseIntError> for VineyardError {
    fn from(error: ParseIntError) -> Self {
        VineyardError::new(StatusCode::IOError, format!("parse int error: {}", error))
    }
}

impl From<ParseFloatError> for VineyardError {
    fn from(error: ParseFloatError) -> Self {
        VineyardError::new(StatusCode::IOError, format!("parse float error: {}", error))
    }
}

impl From<TryFromIntError> for VineyardError {
    fn from(error: TryFromIntError) -> Self {
        VineyardError::new(
            StatusCode::IOError,
            format!("try from int error: {}", error),
        )
    }
}

impl<T> From<PoisonError<T>> for VineyardError {
    fn from(error: PoisonError<T>) -> Self {
        VineyardError::new(StatusCode::Invalid, format!("lock poison error: {}", error))
    }
}

impl From<JSONError> for VineyardError {
    fn from(error: JSONError) -> Self {
        VineyardError::new(StatusCode::IOError, format!("json error: {}", error))
    }
}

pub type Result<T> = std::result::Result<T, VineyardError>;

impl VineyardError {
    pub fn new<T: Into<String>>(code: StatusCode, message: T) -> Self {
        VineyardError {
            code,
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }

    pub fn wrap<T: Into<String>>(self, message: T) -> Self {
        if self.ok() {
            return self;
        }
        VineyardError::new(self.code, format!("{}: {}", self.message, message.into()))
    }

    pub fn invalid<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::Invalid, message.into())
    }

    pub fn key_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::KeyError, message)
    }

    pub fn type_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::TypeError, message)
    }

    pub fn io_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::IOError, message)
    }

    pub fn end_of_file<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::EndOfFile, message)
    }

    pub fn not_implemented<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::NotImplemented, message)
    }

    pub fn assertion_failed<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::AssertionFailed, message)
    }

    pub fn user_input_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::UserInputError, message)
    }

    pub fn object_exists<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ObjectExists, message)
    }

    pub fn object_not_exists<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ObjectNotExists, message)
    }

    pub fn object_sealed<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ObjectSealed, message)
    }

    pub fn object_not_sealed<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ObjectNotSealed, message)
    }

    pub fn object_is_blob<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ObjectIsBlob, message)
    }

    pub fn object_type_error<U: Into<String>, V: Into<String>>(expected: U, actual: V) -> Self {
        VineyardError::new(
            StatusCode::ObjectTypeError,
            format!(
                "expect typename '{}', but got '{}'",
                expected.into(),
                actual.into()
            ),
        )
    }

    pub fn object_spilled(object_id: ObjectID) -> Self {
        VineyardError::new(
            StatusCode::ObjectSpilled,
            format!("object '{}' has already been spilled", object_id),
        )
    }

    pub fn object_not_spilled(object_id: ObjectID) -> Self {
        VineyardError::new(
            StatusCode::ObjectNotSpilled,
            format!("object '{}' hasn't been spilled yet", object_id),
        )
    }

    pub fn meta_tree_invalid<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeInvalid, message)
    }

    pub fn meta_tree_type_invalid<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeTypeInvalid, message)
    }

    pub fn meta_tree_type_not_exists<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeTypeNotExists, message)
    }

    pub fn meta_tree_name_invalid<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeNameInvalid, message)
    }

    pub fn meta_tree_name_not_exists<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeNameNotExists, message)
    }

    pub fn meta_tree_link_invalid<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeLinkInvalid, message)
    }

    pub fn meta_tree_subtree_not_exists<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::MetaTreeSubtreeNotExists, message)
    }

    pub fn vineyard_server_not_ready<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::VineyardServerNotReady, message)
    }

    pub fn arrow_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ArrowError, message)
    }

    pub fn connection_failed<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::ConnectionFailed, message)
    }

    pub fn etcd_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::EtcdError, message)
    }

    pub fn redis_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::RedisError, message)
    }

    pub fn already_stopped<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::AlreadyStopped, message)
    }

    pub fn not_enough_memory<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::NotEnoughMemory, message)
    }

    pub fn stream_drained<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::StreamDrained, message)
    }

    pub fn stream_failed<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::StreamFailed, message)
    }

    pub fn invalid_stream_state<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::InvalidStreamState, message)
    }

    pub fn stream_opened<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::StreamOpened, message)
    }

    pub fn global_object_invalid<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::GlobalObjectInvalid, message)
    }

    pub fn unknown_error<T: Into<String>>(message: T) -> Self {
        VineyardError::new(StatusCode::UnknownError, message)
    }

    pub fn ok(&self) -> bool {
        return self.code == StatusCode::OK;
    }

    pub fn is_invalid(&self) -> bool {
        return self.code == StatusCode::Invalid;
    }

    pub fn is_key_error(&self) -> bool {
        return self.code == StatusCode::KeyError;
    }

    pub fn is_type_error(&self) -> bool {
        return self.code == StatusCode::TypeError;
    }

    pub fn is_io_error(&self) -> bool {
        return self.code == StatusCode::IOError;
    }

    pub fn is_end_of_file(&self) -> bool {
        return self.code == StatusCode::EndOfFile;
    }

    pub fn is_not_implemented(&self) -> bool {
        return self.code == StatusCode::NotImplemented;
    }

    pub fn is_assertion_failed(&self) -> bool {
        return self.code == StatusCode::AssertionFailed;
    }

    pub fn is_user_input_error(&self) -> bool {
        return self.code == StatusCode::UserInputError;
    }

    pub fn is_object_exists(&self) -> bool {
        return self.code == StatusCode::ObjectExists;
    }

    pub fn is_object_not_exists(&self) -> bool {
        return self.code == StatusCode::ObjectNotExists;
    }

    pub fn is_object_sealed(&self) -> bool {
        return self.code == StatusCode::ObjectSealed;
    }

    pub fn is_object_not_sealed(&self) -> bool {
        return self.code == StatusCode::ObjectNotSealed;
    }

    pub fn is_object_is_blob(&self) -> bool {
        return self.code == StatusCode::ObjectIsBlob;
    }

    pub fn is_object_type_error(&self) -> bool {
        return self.code == StatusCode::ObjectTypeError;
    }

    pub fn is_object_spilled(&self) -> bool {
        return self.code == StatusCode::ObjectSpilled;
    }

    pub fn is_object_not_spilled(&self) -> bool {
        return self.code == StatusCode::ObjectNotSpilled;
    }

    pub fn is_meta_tree_invalid(&self) -> bool {
        return self.code == StatusCode::MetaTreeInvalid
            || self.code == StatusCode::MetaTreeNameInvalid
            || self.code == StatusCode::MetaTreeTypeInvalid
            || self.code == StatusCode::MetaTreeLinkInvalid;
    }

    pub fn is_meta_tree_element_not_exists(&self) -> bool {
        return self.code == StatusCode::MetaTreeNameNotExists
            || self.code == StatusCode::MetaTreeTypeNotExists
            || self.code == StatusCode::MetaTreeSubtreeNotExists;
    }

    pub fn is_vineyard_server_not_ready(&self) -> bool {
        return self.code == StatusCode::VineyardServerNotReady;
    }

    pub fn is_arrow_error(&self) -> bool {
        return self.code == StatusCode::ArrowError;
    }

    pub fn is_connection_failed(&self) -> bool {
        return self.code == StatusCode::ConnectionFailed;
    }

    pub fn is_connection_error(&self) -> bool {
        return self.code == StatusCode::ConnectionError;
    }

    pub fn is_etcd_error(&self) -> bool {
        return self.code == StatusCode::EtcdError;
    }

    pub fn is_already_stopped(&self) -> bool {
        return self.code == StatusCode::AlreadyStopped;
    }

    pub fn is_not_enough_memory(&self) -> bool {
        return self.code == StatusCode::NotEnoughMemory;
    }

    pub fn is_stream_drained(&self) -> bool {
        return self.code == StatusCode::StreamDrained;
    }

    pub fn is_stream_failed(&self) -> bool {
        return self.code == StatusCode::StreamFailed;
    }

    pub fn is_invalid_stream_state(&self) -> bool {
        return self.code == StatusCode::InvalidStreamState;
    }

    pub fn is_stream_opened(&self) -> bool {
        return self.code == StatusCode::StreamOpened;
    }

    pub fn is_global_object_invalid(&self) -> bool {
        return self.code == StatusCode::GlobalObjectInvalid;
    }

    pub fn is_unknown_error(&self) -> bool {
        return self.code == StatusCode::UnknownError;
    }

    pub fn code(&self) -> &StatusCode {
        return &self.code;
    }

    pub fn message(&self) -> &String {
        return &self.message;
    }
}

pub fn vineyard_check_ok<T: std::fmt::Debug>(status: Result<T>) {
    if status.is_err() {
        panic!("Error occurs: {:?}.", status)
    }
}

pub fn vineyard_assert<T: Into<String>>(condition: bool, message: T) -> Result<()> {
    if !condition {
        return Err(VineyardError::assertion_failed(format!(
            "assertion failed: {}",
            message.into()
        )));
    }
    return Ok(());
}

pub fn vineyard_assert_typename<U: Into<String> + PartialEq<V>, V: Into<String>>(
    expect: U,
    actual: V,
) -> Result<()> {
    if expect != actual {
        return Err(VineyardError::object_type_error(
            expect.into(),
            actual.into(),
        ));
    }
    return Ok(());
}
