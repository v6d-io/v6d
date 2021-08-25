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
use std::collections::HashMap;

use super::object_meta::ObjectMeta;
use super::object::Object;

pub struct ObjectFactory {}

type ObjectInitializer = Box<Object>;

impl ObjectFactory {
    pub fn create(type_name: &String) -> io::Result<Box<Object>> {
        let known_types = ObjectFactory::get_known_types();


        
        panic!()
    }

    // Question: getKnownTypes?
    pub fn get_known_types() -> HashMap<String, ObjectInitializer> {
        let known_types: HashMap<String, ObjectInitializer> = HashMap::new();
        known_types
    }
}
