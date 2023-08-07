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

use num_traits::ToPrimitive;
use serde_json::Value;

use super::status::*;

pub type JSON = serde_json::Map<String, serde_json::Value>;
pub type JSONResult<T> = serde_json::Result<T>;

pub fn parse_json_object<'a>(root: &'a Value) -> Result<&'a JSON> {
    return root.as_object().ok_or(VineyardError::io_error(
        "incoming message is not a JSON object",
    ));
}

pub fn get_bool(root: &JSON, key: &str) -> Result<bool> {
    return root
        .get(key)
        .ok_or(VineyardError::io_error(format!("key '{}' not found", key)))?
        .as_bool()
        .ok_or(VineyardError::io_error(format!("{} is not a boolean", key)));
}

pub fn get_bool_or(root: &JSON, key: &str, v: bool) -> bool {
    return get_bool(root, key).unwrap_or(v);
}

pub fn get_int<T: From<i64>>(root: &JSON, key: &str) -> Result<T> {
    return root
        .get(key)
        .ok_or(VineyardError::io_error(format!("key '{}' not found", key)))?
        .as_i64()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an integer",
            key
        )))
        .map(|v| v.into());
}

pub fn get_int_or<T: From<i64>>(root: &JSON, key: &str, v: T) -> T {
    return get_int(root, key).unwrap_or(v);
}

pub fn get_isize(root: &JSON, key: &str) -> Result<isize> {
    return get_int::<i64>(root, key)?
        .to_isize()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an isize integer",
            key
        )));
}

pub fn get_isize_or(root: &JSON, key: &str, v: isize) -> isize {
    return get_isize(root, key).unwrap_or(v);
}

pub fn get_int32<T: From<i32>>(root: &JSON, key: &str) -> Result<T> {
    return get_int::<i64>(root, key)?
        .to_i32()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an 32-bit integer",
            key
        )))
        .map(|v| v.into());
}

pub fn get_int32_or<T: From<i32>>(root: &JSON, key: &str, v: T) -> T {
    return get_int32(root, key).unwrap_or(v);
}

pub fn get_uint<T: From<u64>>(root: &JSON, key: &str) -> Result<T> {
    return root
        .get(key)
        .ok_or(VineyardError::io_error(format!("key '{}' not found", key)))?
        .as_u64()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an integer",
            key
        )))
        .map(|v| v.into());
}

pub fn get_uint_or(root: &JSON, key: &str, v: u64) -> u64 {
    return get_uint(root, key).unwrap_or(v);
}

pub fn get_usize(root: &JSON, key: &str) -> Result<usize> {
    return get_uint::<u64>(root, key)?
        .to_usize()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an usize integer",
            key
        )));
}

pub fn get_usize_or(root: &JSON, key: &str, v: usize) -> usize {
    return get_usize(root, key).unwrap_or(v);
}

pub fn get_uint32<T: From<u32>>(root: &JSON, key: &str) -> Result<T> {
    return get_uint::<u64>(root, key)?
        .to_u32()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an unsigned 32-bit integer",
            key
        )))
        .map(|v| v.into());
}

pub fn get_uint32_or<T: From<u32>>(root: &JSON, key: &str, v: T) -> T {
    return get_uint32(root, key).unwrap_or(v);
}

pub fn get_string<'a>(root: &'a JSON, key: &str) -> Result<&'a str> {
    return root
        .get(key)
        .ok_or(VineyardError::io_error(format!("key '{}' not found", key)))?
        .as_str()
        .ok_or(VineyardError::io_error(format!(
            "{} is not an integer",
            key
        )));
}

pub fn get_string_or<'a>(root: &'a JSON, key: &str, v: &'a str) -> &'a str {
    return get_string(root, key).unwrap_or(v);
}
