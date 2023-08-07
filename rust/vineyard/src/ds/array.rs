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

use std::marker::PhantomData;

use crate::client::*;

#[derive(Debug, Clone)]
pub struct Array<T> {
    meta: ObjectMeta,
    size: usize,
    buffer: Box<Blob>,
    phantom: PhantomData<T>,
}

impl<T: TypeName> TypeName for Array<T> {
    fn typename() -> String {
        return format!("vineyard::Array<{}>", T::typename());
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
        vineyard_assert_typename(meta.get_typename()?, &typename::<Array<T>>())?;
        self.meta = meta;

        self.size = self.meta.get_usize("size_")?;
        self.buffer = downcast_object(self.meta.get_member::<Blob>("buffer_")?)?;
        return Ok(());
    }
}

register_vineyard_object!(Array<T: TypeName + 'static>);

impl<T: TypeName + 'static> Array<T> {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut array: Array<T> = Array::default();
        array.construct(meta)?;
        return Ok(Box::new(array));
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
    fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
        if !self.sealed {
            self.set_sealed(true);
        }
        return Ok(());
    }

    fn seal(self: Self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        let buffer = self.buffer.seal(client)?;
        let mut meta = ObjectMeta::from_typename(&typename::<Array<T>>());
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

    pub fn from_vec(client: &mut IPCClient, vec: &Vec<T>) -> Result<Self> {
        let mut builder = ArrayBuilder::new(client, vec.len())?;
        let dest: *mut T = builder.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(vec.as_ptr(), dest, vec.len() * std::mem::size_of::<T>());
        }
        return Ok(builder);
    }

    pub fn from_pointer(client: &mut IPCClient, data: *const T, size: usize) -> Result<Self> {
        let mut builder = ArrayBuilder::new(client, size)?;
        let dest: *mut T = builder.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(data, dest, size * std::mem::size_of::<T>());
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
