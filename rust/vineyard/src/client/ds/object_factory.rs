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
use super::typename::type_name;

pub struct ObjectFactory {}

type ObjectInitializer = Box<dyn Object>;

impl ObjectFactory {
    pub fn register<T>() -> bool {
        let typename = type_name::<T>();
        println!("Register data type: {}", typename);
        let KNOWN_TYPES = ObjectFactory::get_known_types();
        //KNOWN_TYPES.lock().unwrap().insert(typename, Box::new(Object::default()));
        // Question: Casting T::Create to Object
        true
    }

    pub fn create_by_type_name(type_name: &String) -> io::Result<Box<dyn Object>> {
        let known_types = ObjectFactory::get_known_types();
        let known_types = &(**known_types).lock().unwrap();
        let creator = known_types.get(&type_name as &str);
        match creator {
            None => panic!(
                "Failed to create an instance due to the unknown typename: {}",
                type_name
            ),
            Some(initialized_object) => Ok((*initialized_object).clone()), 
            // Question: Add dyn_clone crate
        }
    }

    pub fn create_by_metadata(metadata: ObjectMeta) -> io::Result<Box<dyn Object>> {
        ObjectFactory::create(&metadata.get_type_name(), metadata)
    }

    pub fn create(type_name: &String, metadata: ObjectMeta) -> io::Result<Box<dyn Object>> {
        let known_types = ObjectFactory::get_known_types();
        let known_types = &(**known_types).lock().unwrap();
        let mut creator = known_types.get(&type_name as &str);
        match creator {
            None => panic!(
                "Failed to create an instance due to the unknown typename: {}",
                type_name
            ),
            Some(target) => {
                panic!()
                // Question: Clone or modify the original one? 
                // let mut target = (*target).clone();
                // target.construct(&metadata);
                // return Ok(Box::new(target));
            }
        }
    }

    pub fn factory_ref() -> &'static Mutex<HashMap<&'static str, ObjectInitializer>> {
        return &**ObjectFactory::get_known_types();
    }

    fn get_known_types() -> &'static Arc<Mutex<HashMap<&'static str, ObjectInitializer>>> {
        lazy_static! {
            static ref KNOWN_TYPES: Arc<Mutex<HashMap<&'static str, ObjectInitializer>>> =
                Arc::new(Mutex::new(HashMap::new()));
        }
        &KNOWN_TYPES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_singleton() {
        let KNOWN_TYPES = ObjectFactory::get_known_types();
        println!(
            "Length before insert: {}",
            KNOWN_TYPES.lock().unwrap().len()
        );
        KNOWN_TYPES
            .lock()
            .unwrap()
            .insert("1", Box::new(Object::default()));
        KNOWN_TYPES
            .lock()
            .unwrap()
            .insert("2", Box::new(Object::default()));
        println!(
            "Length after insert: {}",
            ObjectFactory::get_known_types().lock().unwrap().len()
        );
    }
}
