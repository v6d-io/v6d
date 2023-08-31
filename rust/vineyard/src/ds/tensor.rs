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

use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use array::{ArrayRef, OffsetSizeTrait};
use arrow_array as array;
use arrow_array::builder;
use downcast_rs::impl_downcast;
use static_str_ops::*;

use crate::client::*;

use super::arrow::*;

pub trait Tensor: Array {}

impl_downcast!(Tensor);

pub fn downcast_tensor<T: Tensor>(object: Box<dyn Tensor>) -> Result<Box<T>> {
    return object
        .downcast::<T>()
        .map_err(|_| VineyardError::invalid(format!("downcast object to tensor failed",)));
}

pub fn downcast_tensor_ref<T: Tensor>(object: &dyn Tensor) -> Result<&T> {
    return object
        .downcast_ref::<T>()
        .ok_or(VineyardError::invalid(format!(
            "downcast object '{:?}' to tensor failed",
            object.meta().get_typename()?,
        )));
}

pub fn downcast_tensor_rc<T: Tensor>(object: Rc<dyn Tensor>) -> Result<Rc<T>> {
    return object
        .downcast_rc::<T>()
        .map_err(|_| VineyardError::invalid(format!("downcast object to tensor failed",)));
}

#[derive(Debug)]
pub struct NumericTensor<T: NumericType> {
    meta: ObjectMeta,
    shape: Vec<usize>,
    tensor: Arc<TypedArray<T>>,
}

pub type Int8Tensor = NumericTensor<i8>;
pub type UInt8Tensor = NumericTensor<u8>;
pub type Int16Tensor = NumericTensor<i16>;
pub type UInt16Tensor = NumericTensor<u16>;
pub type Int32Tensor = NumericTensor<i32>;
pub type UInt32Tensor = NumericTensor<u32>;
pub type Int64Tensor = NumericTensor<i64>;
pub type UInt64Tensor = NumericTensor<u64>;
pub type Float32Tensor = NumericTensor<f32>;
pub type Float64Tensor = NumericTensor<f64>;

impl<T: TypeName + NumericType> TypeName for NumericTensor<T> {
    fn typename() -> &'static str {
        return staticize(format!("vineyard::Tensor<{}>", T::typename()));
    }
}

impl<T: NumericType + TypeName + 'static> Array for NumericTensor<T> {
    fn array(&self) -> array::ArrayRef {
        return self.tensor.clone();
    }
}

impl<T: NumericType + TypeName + 'static> Tensor for NumericTensor<T> {}

impl<T: NumericType> Default for NumericTensor<T> {
    fn default() -> Self {
        NumericTensor {
            meta: ObjectMeta::default(),
            shape: vec![],
            tensor: Arc::new(TypedArray::<T>::new(vec![].into(), None)),
        }
    }
}

impl<T: TypeName + NumericType + 'static> Object for NumericTensor<T> {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        self.shape = self.meta.get_vector("shape_")?;
        let values: arrow_buffer::ScalarBuffer<_> =
            resolve_scalar_buffer::<T>(&self.meta, "buffer_")?;
        self.tensor = Arc::new(TypedArray::<T>::new(values, None));
        return Ok(());
    }
}

register_vineyard_object!(NumericTensor<T: TypeName + NumericType + 'static>);
register_vineyard_types! {
    Int8Tensor;
    UInt8Tensor;
    Int16Tensor;
    UInt16Tensor;
    Int32Tensor;
    UInt32Tensor;
    Int64Tensor;
    UInt64Tensor;
    Float32Tensor;
    Float64Tensor;
}

impl<T: NumericType + TypeName + 'static> NumericTensor<T> {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut array = Box::<Self>::default();
        array.construct(meta)?;
        return Ok(array);
    }

    pub fn data(&self) -> Arc<TypedArray<T>> {
        return self.tensor.clone();
    }

    pub fn shape(&self) -> &[usize] {
        return &self.shape;
    }

    pub fn len(&self) -> usize {
        return self.shape.iter().product::<usize>();
    }

    pub fn is_empty(&self) -> bool {
        return self.len() == 0;
    }

    pub fn as_slice(&self) -> &[T] {
        return unsafe {
            std::slice::from_raw_parts(
                self.tensor.values().inner().as_ptr() as _,
                self.tensor.len(),
            )
        };
    }
}

impl<T: NumericType> AsRef<TypedArray<T>> for NumericTensor<T> {
    fn as_ref(&self) -> &TypedArray<T> {
        return &self.tensor;
    }
}

pub struct NumericTensorBuilder<T: NumericType> {
    sealed: bool,
    shape: Vec<usize>,
    buffer: BlobWriter,
    phantom: PhantomData<T>,
}

pub type Int8TensorBuilder = NumericTensorBuilder<i8>;
pub type UInt8TensorBuilder = NumericTensorBuilder<u8>;
pub type Int16TensorBuilder = NumericTensorBuilder<i16>;
pub type UInt16TensorBuilder = NumericTensorBuilder<u16>;
pub type Int32TensorBuilder = NumericTensorBuilder<i32>;
pub type UInt32TensorBuilder = NumericTensorBuilder<u32>;
pub type Int64TensorBuilder = NumericTensorBuilder<i64>;
pub type UInt64TensorBuilder = NumericTensorBuilder<u64>;
pub type Float32TensorBuilder = NumericTensorBuilder<f32>;
pub type Float64TensorBuilder = NumericTensorBuilder<f64>;

impl<T: TypeName + NumericType + 'static> ObjectBuilder for NumericTensorBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl<T: TypeName + NumericType + 'static> ObjectBase for NumericTensorBuilder<T> {
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
        let nbytes = self.buffer.len();
        let buffer = self.buffer.seal(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<NumericTensor<T>>());
        meta.add_member("buffer_", buffer)?;
        meta.add_vector("shape_", &self.shape)?;
        meta.set_nbytes(nbytes);
        let metadata = client.create_metadata(&meta)?;
        return NumericTensor::<T>::new_boxed(metadata);
    }
}

impl<T: NumericType> NumericTensorBuilder<T> {
    pub fn new(client: &mut IPCClient, shape: &[usize], array: &TypedArray<T>) -> Result<Self> {
        let buffer = build_scalar_buffer::<T>(client, array.values())?;
        return Ok(NumericTensorBuilder {
            sealed: false,
            shape: shape.to_vec(),
            buffer,
            phantom: PhantomData,
        });
    }

    pub fn new_allocated(client: &mut IPCClient, shape: &[usize]) -> Result<Self> {
        let length = shape.iter().product::<usize>();
        let buffer = client.create_blob(std::mem::size_of::<T>() * length)?;
        return Ok(NumericTensorBuilder {
            sealed: false,
            shape: shape.to_vec(),
            buffer,
            phantom: PhantomData,
        });
    }

    pub fn new_from_array_1d(client: &mut IPCClient, array: &TypedArray<T>) -> Result<Self> {
        return Self::new(client, &[array.len()], array);
    }

    pub fn new_from_builder(
        client: &mut IPCClient,
        shape: &[usize],
        builder: &mut TypedBuilder<T>,
    ) -> Result<Self> {
        let array = builder.finish();
        return Self::new(client, shape, &array);
    }

    pub fn shape(&self) -> &[usize] {
        return &self.shape;
    }

    pub fn len(&self) -> usize {
        return self.shape.iter().product::<usize>();
    }

    pub fn is_empty(&self) -> bool {
        return self.len() == 0;
    }

    pub fn as_slice(&mut self) -> &[T] {
        return unsafe { std::mem::transmute(self.buffer.as_slice()) };
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        return unsafe { std::mem::transmute(self.buffer.as_mut_slice()) };
    }
}

#[derive(Debug)]
pub struct StringTensor {
    meta: ObjectMeta,
    shape: Vec<usize>,
    tensor: Arc<array::GenericStringArray<i64>>,
}

impl Array for StringTensor {
    fn array(&self) -> array::ArrayRef {
        return self.tensor.clone();
    }
}

impl Tensor for StringTensor {}

impl TypeName for StringTensor {
    fn typename() -> &'static str {
        return staticize("vineyard::Tensor<std::string>");
    }
}

impl Default for StringTensor {
    fn default() -> Self {
        StringTensor {
            meta: ObjectMeta::default(),
            shape: vec![],
            tensor: Arc::new(array::GenericStringArray::<i64>::new_null(0)),
        }
    }
}

impl Object for StringTensor {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        self.shape = self.meta.get_vector("shape_")?;
        self.tensor = self.meta.get_member::<LargeStringArray>("buffer_")?.data();
        return Ok(());
    }
}

register_vineyard_object!(StringTensor);

impl StringTensor {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut array = Box::<Self>::default();
        array.construct(meta)?;
        return Ok(array);
    }

    pub fn data(&self) -> Arc<array::GenericStringArray<i64>> {
        return self.tensor.clone();
    }

    pub fn shape(&self) -> &[usize] {
        return &self.shape;
    }

    pub fn len(&self) -> usize {
        return self.shape.iter().product::<usize>();
    }

    pub fn is_empty(&self) -> bool {
        return self.len() == 0;
    }

    pub fn as_slice(&self) -> &[u8] {
        return self.tensor.value_data();
    }

    pub fn as_slice_offsets(&self) -> &[i64] {
        return self.tensor.value_offsets();
    }
}

impl AsRef<array::GenericStringArray<i64>> for StringTensor {
    fn as_ref(&self) -> &array::GenericStringArray<i64> {
        return &self.tensor;
    }
}

pub struct BaseStringTensorBuilder<O: OffsetSizeTrait> {
    sealed: bool,
    shape: Vec<usize>,
    tensor: BaseStringBuilder<O>,
}

pub type StringTensorBuilder = BaseStringTensorBuilder<i32>;
pub type LargeStringTensorBuilder = BaseStringTensorBuilder<i64>;

impl<O: OffsetSizeTrait> ObjectBuilder for BaseStringTensorBuilder<O> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl<O: OffsetSizeTrait> ObjectBase for BaseStringTensorBuilder<O> {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        self.tensor.build(client)?;
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let nbytes = self.tensor.len();
        let tensor = self.tensor.seal(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<StringTensor>());
        meta.add_member("buffer_", tensor)?;
        meta.add_vector("shape_", &self.shape)?;
        meta.add_vector::<i64>("partition_index_", &[-1, -1])?;
        meta.set_nbytes(nbytes);
        let metadata = client.create_metadata(&meta)?;
        return StringTensor::new_boxed(metadata);
    }
}

impl<O: OffsetSizeTrait> BaseStringTensorBuilder<O> {
    pub fn new(
        client: &mut IPCClient,
        shape: &[usize],
        array: &array::GenericStringArray<O>,
    ) -> Result<Self> {
        return Ok(BaseStringTensorBuilder {
            sealed: false,
            shape: shape.to_vec(),
            tensor: BaseStringBuilder::<O>::new(client, array)?,
        });
    }

    pub fn new_from_array_1d(
        client: &mut IPCClient,
        array: &array::GenericStringArray<O>,
    ) -> Result<Self> {
        use array::Array;
        return Self::new(client, &[array.len()], array);
    }

    pub fn new_from_builder(
        client: &mut IPCClient,
        shape: &[usize],
        builder: &mut builder::GenericStringBuilder<O>,
    ) -> Result<Self> {
        let array = builder.finish();
        return Self::new(client, shape, &array);
    }

    pub fn shape(&self) -> &[usize] {
        return &self.shape;
    }

    pub fn len(&self) -> usize {
        return self.shape.iter().product::<usize>();
    }

    pub fn is_empty(&self) -> bool {
        return self.len() == 0;
    }

    pub fn as_slice(&mut self) -> &[u8] {
        return self.tensor.as_slice();
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        return self.tensor.as_mut_slice();
    }

    pub fn as_slice_offsets(&mut self) -> &[O] {
        return self.tensor.as_slice_offsets();
    }

    pub fn as_mut_slice_offsets(&mut self) -> &mut [O] {
        return self.tensor.as_mut_slice_offsets();
    }
}

pub fn downcast_to_tensor(object: Box<dyn Object>) -> Result<Box<dyn Tensor>> {
    macro_rules! downcast {
        ($object: ident, $ty: ty) => {
            |$object| match $object.downcast::<$ty>() {
                Ok(array) => Ok(array),
                Err(original) => Err(original),
            }
        };
    }

    let mut object: std::result::Result<Box<dyn Tensor>, Box<dyn Object>> = Err(object);
    object = object
        .or_else(downcast!(object, Int8Tensor))
        .or_else(downcast!(object, UInt8Tensor))
        .or_else(downcast!(object, Int16Tensor))
        .or_else(downcast!(object, UInt16Tensor))
        .or_else(downcast!(object, Int32Tensor))
        .or_else(downcast!(object, UInt32Tensor))
        .or_else(downcast!(object, Int64Tensor))
        .or_else(downcast!(object, UInt64Tensor))
        .or_else(downcast!(object, Float32Tensor))
        .or_else(downcast!(object, Float64Tensor))
        .or_else(downcast!(object, StringTensor))
        .or_else(downcast!(object, StringTensor));

    match object {
        Ok(array) => return Ok(array),
        Err(object) => {
            return Err(VineyardError::invalid(format!(
                "downcast object to tensor failed, object type is: '{}'",
                object.meta().get_typename()?,
            )))
        }
    };
}

pub fn build_tensor(client: &mut IPCClient, array: ArrayRef) -> Result<Box<dyn Object>> {
    macro_rules! build {
        ($array: ident, $array_ty: ty, $builder_ty: ty) => {
            |$array| match $array.as_any().downcast_ref::<$array_ty>() {
                Some(array) => match <$builder_ty>::new_from_array_1d(client, array) {
                    Ok(builder) => match builder.seal(client) {
                        Ok(object) => Ok(object),
                        Err(_) => Err(array as &dyn array::Array),
                    },
                    Err(_) => Err(array as &dyn array::Array),
                },
                None => Err($array),
            }
        };
    }

    let mut array: std::result::Result<Box<dyn Object>, &dyn array::Array> = Err(array.as_ref());
    array = array
        .or_else(build!(array, array::Int8Array, Int8TensorBuilder))
        .or_else(build!(array, array::UInt8Array, UInt8TensorBuilder))
        .or_else(build!(array, array::Int16Array, Int16TensorBuilder))
        .or_else(build!(array, array::UInt16Array, UInt16TensorBuilder))
        .or_else(build!(array, array::Int32Array, Int32TensorBuilder))
        .or_else(build!(array, array::UInt32Array, UInt32TensorBuilder))
        .or_else(build!(array, array::Int64Array, Int64TensorBuilder))
        .or_else(build!(array, array::UInt64Array, UInt64TensorBuilder))
        .or_else(build!(array, array::Float32Array, Float32TensorBuilder))
        .or_else(build!(array, array::Float64Array, Float64TensorBuilder))
        .or_else(build!(array, array::StringArray, StringTensorBuilder))
        .or_else(build!(
            array,
            array::LargeStringArray,
            LargeStringTensorBuilder
        ));

    match array {
        Ok(builder) => return Ok(builder),
        Err(array) => {
            return Err(VineyardError::invalid(format!(
                "build array failed, array type is: '{}'",
                array.data_type(),
            )))
        }
    };
}
