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

use rand::Rng;

pub type ObjectID = u64;
pub type InstanceID = u64;
pub type Signature = u64;

use super::status::*;

pub fn empty_blob_id() -> ObjectID {
    0x8000000000000000u64
}

pub fn generate_object_id() -> ObjectID {
    let mut rng = rand::thread_rng();
    0x7FFFFFFFFFFFFFFFu64 & rng.gen::<u64>()
}

pub fn generate_signature() -> Signature {
    let mut rng = rand::thread_rng();
    0x7FFFFFFFFFFFFFFFu64 & rng.gen::<u64>()
}

pub fn is_blob(id: ObjectID) -> bool {
    return !(id & 0x8000000000000000u64 == 0);
}

pub fn object_id_from_string(s: &str) -> Result<ObjectID> {
    if s.len() < 2 {
        return Err(VineyardError::invalid(format!(
            "invalid object id: '{}'",
            s
        )));
    }
    return Ok(ObjectID::from_str_radix(&s[1..], 16)?);
}

pub fn object_id_to_string(id: ObjectID) -> String {
    format!("o{:x}", id)
}

pub fn signature_to_string(id: ObjectID) -> String {
    format!("s{:x}", id)
}

pub fn invalid_object_id() -> ObjectID {
    ObjectID::MAX
}

pub fn invalid_signature() -> Signature {
    Signature::MAX
}

pub fn unspecified_instance_id() -> InstanceID {
    InstanceID::MAX
}
