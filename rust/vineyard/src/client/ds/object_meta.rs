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

use std::rc::{Rc, Weak};
use std::ops;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use super::{ObjectID, InstanceID, Signature};
use super::{Client, ClientKind};
use super::blob::BufferSet;
use super::object::Object;

pub struct ObjectMeta {
    client: Weak<ClientKind>, // Since W<T> doesn't have T:?Sized for Weak<dyn Client>  
    meta: Value,
    buffer_set: Rc<BufferSet>, 
    incomplete: bool,
    force_local: bool,
}

impl Default for ObjectMeta {
    fn default() -> Self {
        ObjectMeta {
            client: Weak::new(),
            meta: json!({}),
            buffer_set: Rc::new(BufferSet{}),
            incomplete: false,
            force_local: false,
        }
    }
}

impl ObjectMeta {
    fn from(other: &ObjectMeta) -> ObjectMeta {
        ObjectMeta {
            client: other.client.clone(),
            meta: other.meta.clone(),
            buffer_set: Rc::clone(&other.buffer_set),
            incomplete: other.incomplete,
            force_local: other.force_local,
        }
    }

    fn set_client(&mut self, client: Weak<ClientKind>) {
        self.client = client;
    }

    fn get_client(&self) -> Weak<ClientKind> {
        self.client.clone()
    }

    fn set_id(&mut self, id: ObjectID) {
        self.meta = serde_json::from_str(&id.to_string()).unwrap();
    }

    fn get_id(&self) -> ObjectID {
        self.meta["id"].as_u64().unwrap()
    }

    fn get_signature(&self) -> Signature {
        self.meta["signature"].as_u64().unwrap()
    }

    fn reset_signature() {

    }

    fn set_global() {}

    fn is_global() {}

    // fn set_type_name() {}

    // fn get_type_name() {}

    // fn set_n_bytes() {}

    // fn get_n_bytes() {}

    // fn get_instance_id() {}

    // fn is_local() {}

    // fn has_key() {}

    // fn reset_key() {}

    // fn add_key_value() {}

    // fn get_key_value() {}

    // fn add_member() {}

    // fn get_member() {}

    // fn get_member_meta() {}

    // fn get_buffer() {}

    // fn set_buffer() {}
}
