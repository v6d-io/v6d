use std::cell::RefCell;
use std::collections::HashMap;
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
use std::io;
use std::sync::{Arc, Mutex};

use lazy_static::lazy_static;

use super::object::Object;
use super::object_meta::ObjectMeta;

pub struct ObjectFactory {}

type ObjectInitializer = Box<Object>;

impl ObjectFactory {
    pub fn create_by_type_name(type_name: &String) -> io::Result<Box<Object>> {
        //let known_types = ObjectFactory::get_known_types();

        panic!()
    }

    pub fn get_known_types() -> Mutex<HashMap<String, ObjectInitializer>> {
        panic!()
    }
}

lazy_static! {
    // static ref KNOWN_TYPES: Arc<Mutex<HashMap<&'static str, ObjectInitializer>>>
    //     = Arc::new(Mutex::new(HashMap::new()));
    // static ref KNOWN_TYPES1: HashMap<&'static str, ObjectInitializer>
    //     = HashMap::new();
}
