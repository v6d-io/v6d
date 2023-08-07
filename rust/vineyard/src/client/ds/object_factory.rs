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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ctor::ctor;

use crate::common::util::status::*;
use crate::common::util::typename::typename;

use super::object::{Create, Object};
use super::object_meta::ObjectMeta;

pub struct ObjectFactory {}

type ObjectInitializer = fn() -> Box<dyn Object>;

#[ctor]
static KNOWN_TYPES: Arc<Mutex<HashMap<&'static str, ObjectInitializer>>> =
    Arc::new(Mutex::new(HashMap::new()));

impl ObjectFactory {
    pub fn register<T: Create>() -> Result<bool> {
        let typename = typename::<T>();
        let known_types = ObjectFactory::get_known_types();
        let closure: ObjectInitializer = || T::create();
        let inserted = known_types.lock()?.insert(typename, closure);
        return Ok(inserted.is_none());
    }

    pub fn create(typename: &str) -> Result<Box<dyn Object>> {
        let known_types = ObjectFactory::get_known_types();
        return match known_types.lock()?.get(typename) {
            None => Err(VineyardError::invalid(format!(
                "Failed to create an instance due to the unknown typename: {}",
                typename
            ))),
            Some(initializer) => {
                return Ok((*initializer)());
            }
        };
    }

    pub fn create_from_metadata(metadata: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut object = ObjectFactory::create(metadata.get_typename()?)?;
        object.construct(metadata)?;
        return Ok(object);
    }

    pub fn create_from_typename_and_metadata(
        typename: &str,
        metadata: ObjectMeta,
    ) -> Result<Box<dyn Object>> {
        let mut object = ObjectFactory::create(typename)?;
        object.construct(metadata)?;
        return Ok(object);
    }

    pub fn factory_ref() -> &'static Mutex<HashMap<&'static str, ObjectInitializer>> {
        return ObjectFactory::get_known_types();
    }

    fn get_known_types() -> &'static Arc<Mutex<HashMap<&'static str, ObjectInitializer>>> {
        lazy_static! {
            static ref KNOWN_TYPES: Arc<Mutex<HashMap<&'static str, ObjectInitializer>>> =
                Arc::new(Mutex::new(HashMap::new()));
        }
        return &KNOWN_TYPES;
    }
}
