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

use serde_derive::{Deserialize, Serialize};

use super::super::util::json::*;
use super::super::util::status::*;
use super::super::util::uuid::ObjectID;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Payload {
    pub object_id: ObjectID,
    pub store_fd: i32,
    pub arena_fd: i32,
    pub data_offset: isize,
    pub data_size: usize,
    pub map_size: usize,
    pub ref_cnf: i64,
    pub is_sealed: bool,
    pub is_owner: bool,
    pub is_spilled: bool,
    pub is_gpu: bool,
}

impl Payload {
    pub fn empty() -> Payload {
        return Default::default();
    }

    pub fn from_json(root: &JSON) -> Result<Self> {
        return Ok(Payload {
            object_id: get_uint(root, "object_id")?,
            store_fd: get_int32_or(root, "store_fd", -1 as i32),
            arena_fd: get_int32_or(root, "arena_fd", -1 as i32),
            data_offset: get_isize_or(root, "data_offset", 0 as isize),
            data_size: get_usize_or(root, "data_size", 0 as usize),
            map_size: get_usize_or(root, "map_size", 0 as usize),
            ref_cnf: get_int_or(root, "ref_count", -1),
            is_sealed: get_bool_or(root, "is_sealed", false),
            is_owner: get_bool_or(root, "is_owner", false),
            is_spilled: get_bool_or(root, "is_spilled", false),
            is_gpu: get_bool_or(root, "is_gpu", false),
        });
    }
}
