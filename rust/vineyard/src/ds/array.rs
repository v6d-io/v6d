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

use std::convert::AsRef;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use static_str_ops::*;

use crate::client::*;

#[derive(Debug)]
pub struct Array<T> {
    meta: ObjectMeta,
    size: usize,
    buffer: Box<Blob>,
    phantom: PhantomData<T>,
}

impl<T: TypeName> TypeName for Array<T> {
    fn typename() -> &'static str {
        return staticize(format!("vineyard::Array<{}>", T::typename()));
    }
}

impl<T> Default for Array<T> {
    fn default() -> Self {
        Array {
            meta: ObjectMeta::default(),
            size: 0,
            buffer: Box::new(Blob::default()),
            phantom: PhantomData,
        }
    }
}

impl<T: TypeName + 'static> Object for Array<T> {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;

        self.size = self.meta.get_usize("size_")?;
        self.buffer = downcast_object(self.meta.get_member::<Blob>("buffer_")?)?;
        return Ok(());
    }
}

register_vineyard_object!(Array<T: TypeName + 'static>);
register_vineyard_types! {
    Array<i8>;
    Array<u8>;
    Array<i16>;
    Array<u16>;
    Array<i32>;
    Array<u32>;
    Array<i64>;
    Array<u64>;
    Array<f32>;
    Array<f64>;
}

impl<T: TypeName + 'static> Array<T> {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut array = Box::<Self>::default();
        array.construct(meta)?;
        return Ok(array);
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr(&self) -> *const T {
        let ptr = self.buffer.as_ptr_unchecked();
        return ptr as *const T;
    }

    pub fn as_slice(&self) -> &[T] {
        let ptr = self.buffer.as_ptr_unchecked();
        return unsafe { std::slice::from_raw_parts(ptr as *const T, self.size) };
    }
}

impl<T: TypeName + 'static> Deref for Array<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return self.as_slice();
    }
}

impl<T: TypeName + 'static> AsRef<[T]> for Array<T> {
    fn as_ref(&self) -> &[T] {
        return self.as_slice();
    }
}

pub struct ArrayBuilder<T> {
    sealed: bool,
    size: usize,
    buffer: BlobWriter,
    phantom: PhantomData<T>,
}

impl<T: TypeName + 'static> ObjectBuilder for ArrayBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl<T: TypeName + 'static> ObjectBase for ArrayBuilder<T> {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        self.buffer.build(client)?;
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let buffer = self.buffer.seal(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<Array<T>>());
        meta.add_member("buffer_", buffer)?;
        meta.add_usize("size_", self.size);
        meta.set_nbytes(self.size * std::mem::size_of::<T>());
        let metadata = client.create_metadata(&meta)?;
        return Array::<T>::new_boxed(metadata);
    }
}

impl<T: TypeName + 'static> ArrayBuilder<T> {
    pub fn new(client: &mut IPCClient, size: usize) -> Result<Self> {
        let buffer = client.create_blob(size * std::mem::size_of::<T>())?;
        let builder = ArrayBuilder {
            sealed: false,
            size,
            buffer,
            phantom: PhantomData,
        };
        return Ok(builder);
    }

    pub fn from_vec(client: &mut IPCClient, vec: &[T]) -> Result<Self> {
        let mut builder = ArrayBuilder::new(client, vec.len())?;
        let dest: *mut T = builder.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(vec.as_ptr(), dest, vec.len());
        }
        return Ok(builder);
    }

    pub fn from_pointer(client: &mut IPCClient, data: *const T, size: usize) -> Result<Self> {
        let mut builder = ArrayBuilder::new(client, size)?;
        let dest: *mut T = builder.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(data, dest, size);
        }
        return Ok(builder);
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr(&self) -> *const T {
        return self.buffer.as_ptr() as *const T;
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        return self.buffer.as_mut_ptr() as *mut T;
    }

    pub fn as_slice(&self) -> &[T] {
        return unsafe { std::slice::from_raw_parts(self.as_ptr(), self.size) };
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        return unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.size) };
    }
}

impl<T: TypeName + 'static> Deref for ArrayBuilder<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return self.as_slice();
    }
}

impl<T: TypeName + 'static> DerefMut for ArrayBuilder<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return self.as_mut_slice();
    }
}

impl<T: TypeName + 'static> AsRef<[T]> for ArrayBuilder<T> {
    fn as_ref(&self) -> &[T] {
        return self.as_slice();
    }
}
