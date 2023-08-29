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

use std::any::Any;
use std::rc::Rc;

use downcast_rs::impl_downcast;
use downcast_rs::Downcast;

use crate::common::util::status::*;
use crate::common::util::typename::*;
use crate::common::util::uuid::*;

use super::super::IPCClient;
use super::object_meta::ObjectMeta;

/// Re-export the gensym and ctor symbol to avoid introducing new
/// dependencies (gensym, ctor) in callers.
pub use ctor::ctor;
pub use gensym::gensym;

pub trait Create: TypeName {
    fn create() -> Box<dyn Object>;
}

pub trait ObjectBase: Downcast {
    fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
        return Ok(());
    }

    fn seal(self, client: &mut IPCClient) -> Result<Box<dyn Object>>;
}

impl_downcast!(ObjectBase);

pub trait ObjectMetaAttr {
    fn id(&self) -> ObjectID {
        self.meta().get_id()
    }

    /// Return a reference to the meta data of the object.
    fn meta(&self) -> &ObjectMeta;

    /// Return a move of the inner metadata.
    fn metadata(self) -> ObjectMeta;

    fn nbytes(&self) -> usize {
        self.meta().get_nbytes()
    }

    fn is_local(&self) -> bool {
        self.meta().is_local()
    }

    fn is_global(&self) -> bool {
        self.meta().is_global()
    }
}

pub trait Object: ObjectBase + ObjectMetaAttr {
    fn as_any(&'_ self) -> &'_ dyn Any
    where
        Self: Sized + 'static,
    {
        self
    }

    fn construct(&mut self, meta: ObjectMeta) -> Result<()>;
}

impl_downcast!(Object);

pub fn downcast<T: Object + TypeName>(
    object: Box<dyn Object>,
) -> std::result::Result<Box<T>, Box<dyn Object>> {
    return object.downcast::<T>();
}

pub fn downcast_object<T: Object + TypeName>(object: Box<dyn Object>) -> Result<Box<T>> {
    return object.downcast::<T>().map_err(|_| {
        VineyardError::invalid(format!(
            "downcast object to type '{}' failed",
            T::typename()
        ))
    });
}

pub fn downcast_object_ref<T: Object + TypeName>(object: &dyn Object) -> Result<&T> {
    return object
        .downcast_ref::<T>()
        .ok_or(VineyardError::invalid(format!(
            "downcast object to type '{}' failed",
            T::typename()
        )));
}

pub fn downcast_rc<T: Object + TypeName>(
    object: Rc<dyn Object>,
) -> std::result::Result<Rc<T>, Rc<dyn Object>> {
    return object.downcast_rc::<T>();
}

pub fn downcast_object_rc<T: Object + TypeName>(object: Rc<dyn Object>) -> Result<Rc<T>> {
    return object.downcast_rc::<T>().map_err(|_| {
        VineyardError::invalid(format!(
            "downcast object to type '{}' failed",
            T::typename()
        ))
    });
}

pub fn downcast_object_mut<T: Object + TypeName>(object: &mut dyn Object) -> Result<&mut T> {
    return object
        .downcast_mut::<T>()
        .ok_or(VineyardError::invalid(format!(
            "downcast object to type '{}' failed",
            T::typename()
        )));
}

pub trait GlobalObject {}

pub trait ObjectBuilder: ObjectBase {
    fn sealed(&self) -> bool;

    fn set_sealed(&mut self, sealed: bool);

    fn ensure_not_sealed(&mut self) -> Result<()> {
        return vineyard_assert(!self.sealed(), "The builder has already been sealed");
    }
}

impl_downcast!(ObjectBuilder);

#[doc(hidden)]
#[macro_export]
macro_rules! register_vineyard_type_impl {
    ($gensym:ident, $type:ty) => {
        #[$crate::client::ds::object::ctor]
        static $gensym: Result<bool> = ObjectFactory::register::<$type>();
    };
}

#[macro_export]
macro_rules! register_vineyard_type {
    ($type:ty) => {
        $crate::client::ds::object::gensym! { $crate::client::ds::object::register_vineyard_type_impl!{ $type } }
    }
}

#[macro_export]
macro_rules! register_vineyard_types {
    {
        $( $type:ty; )+
    } => {
        $(
            $crate::client::ds::object::register_vineyard_type!($type);
        )+
    };
}

#[macro_export]
macro_rules! register_vineyard_object {
    // match when no type parameters are present
    ($t:tt) => {
        $crate::client::ds::object::register_vineyard_type!($t);

        impl Create for $t {
            fn create() -> Box<dyn Object> {
                return Box::new(Self::default());
            }
        }

        impl ObjectMetaAttr for $t {
            fn meta(&self) -> &ObjectMeta {
                return &self.meta;
            }

            fn metadata(self) -> ObjectMeta {
                return self.meta;
            }
        }

        impl ObjectBase for $t {
            fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
                return Ok(());
            }

            fn seal(self: Self, _client: &mut IPCClient) -> Result<Box<dyn Object>>
            {
                return Ok(Box::new(self));
            }
        }
    };
    // this evil monstrosity matches <A, B: T, C: S + T + 'a>
    ($t:ident < $( $N:ident $(: $b0:ident $(+$bs:ident)* $(+$lts:lifetime)* )? ),* >) =>
    {
        impl< $( $N $(: $b0 $(+$bs)* $(+$lts)* )? ),* > Create for $t< $( $N ),* > {
            fn create() -> Box<dyn Object> {
                return Box::new(Self::default());
            }
        }

        impl< $( $N $(: $b0 $(+$bs)* $(+$lts)* )? ),* > ObjectMetaAttr for $t< $( $N ),* > {
            fn meta(&self) -> &ObjectMeta {
                return &self.meta;
            }

            fn metadata(self) -> ObjectMeta {
                return self.meta;
            }
        }

        impl< $( $N $(: $b0 $(+$bs)* $(+$lts)* )? ),* > ObjectBase for $t< $( $N ),* > {
            fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
                return Ok(());
            }

            fn seal(self: Self, _client: &mut IPCClient) -> Result<Box<dyn Object>>
            {
                return Ok(Box::new(self));
            }
        }
    };
}

pub use register_vineyard_object;
pub use register_vineyard_type;
pub use register_vineyard_type_impl;
pub use register_vineyard_types;
