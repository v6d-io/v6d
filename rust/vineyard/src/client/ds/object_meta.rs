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

use std::rc::Rc;
use std::ops;

use serde::{Deserialize, Serialize};
use serde_json::Result as JsonResult;
use serde_json::{json, Value};

use crate::client::Object;

use super::{Client, IPCClient};
use super::blob::BufferSet;

pub struct ObjectMeta {
    client: Option<Rc<dyn Client>>, // Question: Rc or Box or raw
    meta: Value,
    buffer_set: Option<Rc<BufferSet>>, // Question: or Rc<Option<BufferSet>>
    incomplete: bool,
    force_local: bool,
}

impl Default for ObjectMeta {
    fn default() -> Self {
        ObjectMeta {
            client: None,
            meta: json!({}),
            buffer_set: None,
            incomplete: false,
            force_local: false,
        }
    }
}

impl ObjectMeta {
    fn from(other: &ObjectMeta) -> ObjectMeta {
        ObjectMeta {
            client: Some(Rc::clone(&other.client.as_ref().unwrap())),
            meta: other.meta.clone(),
            buffer_set: Some(Rc::clone(&other.buffer_set.as_ref().unwrap())),
            incomplete: other.incomplete,
            force_local: other.force_local,
        }
    }

    fn set_client(&mut self, client: Rc<dyn Client>) {
        self.client = Some(client);
    }

    fn get_client(&self) -> &Rc<dyn Client> {
        match &self.client {
            Some(client) => &client,
            None => panic!("The object has no client"),
        }
    }

    // fn set_id() {}

    // fn get_id() {}

    // fn get_signature() {}

    // fn set_global() {}

    // fn is_global() {}

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
