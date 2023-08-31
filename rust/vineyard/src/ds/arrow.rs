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
use std::rc::Rc;
use std::sync::Arc;

use arrow_array as array;
use arrow_array::builder;
use arrow_array::builder::GenericStringBuilder;
use arrow_array::{ArrayRef, GenericStringArray, OffsetSizeTrait};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema as schema;
use arrow_schema::ArrowError;
use downcast_rs::impl_downcast;
use itertools::izip;
use serde_json::{json, Value};
use static_str_ops::*;

use super::arrow_utils::*;
use crate::client::*;

impl From<ArrowError> for VineyardError {
    fn from(error: ArrowError) -> Self {
        VineyardError::new(StatusCode::ArrowError, format!("{}", error))
    }
}

pub trait Array: Object {
    fn array(&self) -> array::ArrayRef;
}

impl_downcast!(Array);

pub fn downcast_array<T: Array>(object: Box<dyn Array>) -> Result<Box<T>> {
    return object
        .downcast::<T>()
        .map_err(|_| VineyardError::invalid(format!("downcast object to array failed",)));
}

pub fn downcast_array_ref<T: Array>(object: &dyn Array) -> Result<&T> {
    return object
        .downcast_ref::<T>()
        .ok_or(VineyardError::invalid(format!(
            "downcast object '{:?}' to array failed",
            object.meta().get_typename()?,
        )));
}

pub fn downcast_array_rc<T: Array>(object: Rc<dyn Array>) -> Result<Rc<T>> {
    return object
        .downcast_rc::<T>()
        .map_err(|_| VineyardError::invalid(format!("downcast object to array failed",)));
}

pub trait NumericType = ToArrowType where <Self as ToArrowType>::Type: array::ArrowPrimitiveType;

pub type TypedBuffer<T> =
    ScalarBuffer<<<T as ToArrowType>::Type as array::ArrowPrimitiveType>::Native>;

pub type TypedArray<T> = array::PrimitiveArray<<T as ToArrowType>::Type>;
pub type TypedBuilder<T> = builder::PrimitiveBuilder<<T as ToArrowType>::Type>;

#[derive(Debug)]
pub struct NumericArray<T: NumericType> {
    meta: ObjectMeta,
    array: Arc<TypedArray<T>>,
}

impl<T: NumericType + TypeName + 'static> Array for NumericArray<T> {
    fn array(&self) -> array::ArrayRef {
        return self.array.clone();
    }
}

pub type Int8Array = NumericArray<i8>;
pub type UInt8Array = NumericArray<u8>;
pub type Int16Array = NumericArray<i16>;
pub type UInt16Array = NumericArray<u16>;
pub type Int32Array = NumericArray<i32>;
pub type UInt32Array = NumericArray<u32>;
pub type Int64Array = NumericArray<i64>;
pub type UInt64Array = NumericArray<u64>;
pub type Float32Array = NumericArray<f32>;
pub type Float64Array = NumericArray<f64>;

impl<T: TypeName + NumericType> TypeName for NumericArray<T> {
    fn typename() -> &'static str {
        return staticize(format!("vineyard::NumericArray<{}>", T::typename()));
    }
}

impl<T: NumericType> Default for NumericArray<T> {
    fn default() -> Self {
        NumericArray {
            meta: ObjectMeta::default(),
            array: Arc::new(TypedArray::<T>::new(vec![].into(), None)),
        }
    }
}

impl<T: TypeName + NumericType + 'static> Object for NumericArray<T> {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        let values = resolve_scalar_buffer::<T>(&self.meta, "buffer_")?;
        let nulls = resolve_null_bitmap_buffer(&self.meta, "null_bitmap_")?;
        self.array = Arc::new(TypedArray::<T>::new(values, nulls));
        return Ok(());
    }
}

register_vineyard_object!(NumericArray<T: TypeName + NumericType + 'static>);
register_vineyard_types! {
    Int8Array;
    UInt8Array;
    Int16Array;
    UInt16Array;
    Int32Array;
    UInt32Array;
    Int64Array;
    UInt64Array;
    Float32Array;
    Float64Array;
}

impl<T: NumericType + TypeName + 'static> NumericArray<T> {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut array = Box::<Self>::default();
        array.construct(meta)?;
        return Ok(array);
    }

    pub fn data(&self) -> Arc<TypedArray<T>> {
        return self.array.clone();
    }

    pub fn len(&self) -> usize {
        return self.array.len();
    }

    pub fn is_empty(&self) -> bool {
        return self.array.is_empty();
    }

    pub fn as_slice(&self) -> &[T] {
        return unsafe {
            std::slice::from_raw_parts(self.array.values().inner().as_ptr() as _, self.len())
        };
    }
}

impl<T: NumericType> AsRef<TypedArray<T>> for NumericArray<T> {
    fn as_ref(&self) -> &TypedArray<T> {
        return &self.array;
    }
}

pub struct NumericBuilder<T: NumericType> {
    sealed: bool,
    length: usize,
    offset: usize,
    null_count: usize,
    buffer: BlobWriter,
    null_bitmap: Option<BlobWriter>,
    phantom: PhantomData<T>,
}

pub type Int8Builder = NumericBuilder<i8>;
pub type UInt8Builder = NumericBuilder<u8>;
pub type Int16Builder = NumericBuilder<i16>;
pub type UInt16Builder = NumericBuilder<u16>;
pub type Int32Builder = NumericBuilder<i32>;
pub type UInt32Builder = NumericBuilder<u32>;
pub type Int64Builder = NumericBuilder<i64>;
pub type UInt64Builder = NumericBuilder<u64>;
pub type Float32Builder = NumericBuilder<f32>;
pub type Float64Builder = NumericBuilder<f64>;

impl<T: TypeName + NumericType + 'static> ObjectBuilder for NumericBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl<T: TypeName + NumericType + 'static> ObjectBase for NumericBuilder<T> {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        self.buffer.build(client)?;
        if let Some(ref mut null_bitmap) = self.null_bitmap {
            null_bitmap.build(client)?;
        }
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let mut nbytes = self.buffer.len();
        let buffer = self.buffer.seal(client)?;
        let null_bitmap = match self.null_bitmap {
            None => None,
            Some(null_bitmap) => {
                nbytes += null_bitmap.len();
                Some(null_bitmap.seal(client)?)
            }
        };
        let mut meta = ObjectMeta::new_from_typename(typename::<NumericArray<T>>());
        meta.add_member("buffer_", buffer)?;
        if let Some(null_bitmap) = null_bitmap {
            meta.add_member("null_bitmap_", null_bitmap)?;
        } else {
            meta.add_member("null_bitmap_", Blob::empty(client)?)?;
        }
        meta.add_usize("length_", self.length);
        meta.add_usize("offset_", self.offset);
        meta.add_usize("null_count_", self.null_count);
        meta.set_nbytes(nbytes);
        let metadata = client.create_metadata(&meta)?;
        return NumericArray::<T>::new_boxed(metadata);
    }
}

impl<T: NumericType> NumericBuilder<T> {
    pub fn new(client: &mut IPCClient, array: &TypedArray<T>) -> Result<Self> {
        use arrow_array::Array;

        let buffer = build_scalar_buffer::<T>(client, array.values())?;
        let null_bitmap = build_null_bitmap_buffer(client, array.nulls())?;
        return Ok(NumericBuilder {
            sealed: false,
            length: array.len(),
            offset: 0,
            null_count: array.null_count(),
            buffer,
            null_bitmap,
            phantom: PhantomData,
        });
    }

    pub fn new_allocated(client: &mut IPCClient, length: usize) -> Result<Self> {
        let buffer = client.create_blob(std::mem::size_of::<T>() * length)?;
        return Ok(NumericBuilder {
            sealed: false,
            length,
            offset: 0,
            null_count: 0,
            buffer,
            null_bitmap: None,
            phantom: PhantomData,
        });
    }

    pub fn new_from_builder(client: &mut IPCClient, builder: &mut TypedBuilder<T>) -> Result<Self> {
        let array = builder.finish();
        return Self::new(client, &array);
    }

    pub fn len(&self) -> usize {
        return self.length;
    }

    pub fn is_empty(&self) -> bool {
        return self.length == 0;
    }

    pub fn offset(&self) -> usize {
        return self.offset;
    }

    pub fn null_count(&self) -> usize {
        return self.null_count;
    }

    pub fn as_slice(&mut self) -> &[T] {
        return unsafe { std::mem::transmute(self.buffer.as_slice()) };
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        return unsafe { std::mem::transmute(self.buffer.as_mut_slice()) };
    }
}

#[derive(Debug)]
pub struct BaseStringArray<O: OffsetSizeTrait> {
    meta: ObjectMeta,
    array: Arc<array::GenericStringArray<O>>,
}

impl<O: OffsetSizeTrait> Array for BaseStringArray<O> {
    fn array(&self) -> array::ArrayRef {
        return self.array.clone();
    }
}

pub type StringArray = BaseStringArray<i32>;
pub type LargeStringArray = BaseStringArray<i64>;

impl<O: OffsetSizeTrait> TypeName for BaseStringArray<O> {
    fn typename() -> &'static str {
        if std::mem::size_of::<O>() == 4 {
            return staticize("vineyard::BaseBinaryArray<arrow::StringArray>");
        } else {
            return staticize("vineyard::BaseBinaryArray<arrow::LargeStringArray>");
        }
    }
}

impl<O: OffsetSizeTrait> Default for BaseStringArray<O> {
    fn default() -> Self {
        BaseStringArray {
            meta: ObjectMeta::default(),
            array: Arc::new(array::GenericStringArray::<O>::new_null(0)),
        }
    }
}

impl<O: OffsetSizeTrait> Object for BaseStringArray<O> {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        let values = resolve_buffer(&self.meta, "buffer_data_")?;
        let offsets = resolve_offsets_buffer::<O>(&self.meta, "buffer_offsets_")?;
        let nulls = resolve_null_bitmap_buffer(&self.meta, "null_bitmap_")?;
        self.array = Arc::new(array::GenericStringArray::<O>::new(offsets, values, nulls));
        return Ok(());
    }
}

register_vineyard_object!(BaseStringArray<O: OffsetSizeTrait>);
register_vineyard_types! {
    StringArray;
    LargeStringArray;
}

impl<O: OffsetSizeTrait> BaseStringArray<O> {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut array = Box::<Self>::default();
        array.construct(meta)?;
        return Ok(array);
    }

    pub fn data(&self) -> Arc<array::GenericStringArray<O>> {
        return self.array.clone();
    }

    pub fn len(&self) -> usize {
        use arrow_array::Array;
        return self.array.len();
    }

    pub fn is_empty(&self) -> bool {
        use arrow_array::Array;
        return self.array.is_empty();
    }

    pub fn as_slice(&self) -> &[u8] {
        return self.array.value_data();
    }

    pub fn as_slice_offsets(&self) -> &[O] {
        return self.array.value_offsets();
    }
}

impl<O: OffsetSizeTrait> AsRef<array::GenericStringArray<O>> for BaseStringArray<O> {
    fn as_ref(&self) -> &array::GenericStringArray<O> {
        return &self.array;
    }
}

pub struct BaseStringBuilder<O: OffsetSizeTrait> {
    sealed: bool,
    length: usize,
    offset: usize,
    null_count: usize,
    value_data: BlobWriter,
    value_offsets: BlobWriter,
    null_bitmap: Option<BlobWriter>,
    phantom: PhantomData<O>,
}

pub type StringBuilder = BaseStringBuilder<i32>;
pub type LargeStringBuilder = BaseStringBuilder<i64>;

impl<O: OffsetSizeTrait> ObjectBuilder for BaseStringBuilder<O> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl<O: OffsetSizeTrait> ObjectBase for BaseStringBuilder<O> {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        self.value_data.build(client)?;
        self.value_offsets.build(client)?;
        if let Some(ref mut null_bitmap) = self.null_bitmap {
            null_bitmap.build(client)?;
        }
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let mut nbytes = self.value_data.len();
        let value_data = self.value_data.seal(client)?;
        nbytes += self.value_offsets.len();
        let value_offsets = self.value_offsets.seal(client)?;
        let null_bitmap = match self.null_bitmap {
            None => None,
            Some(null_bitmap) => {
                nbytes += null_bitmap.len();
                Some(null_bitmap.seal(client)?)
            }
        };
        let mut meta = ObjectMeta::new_from_typename(typename::<BaseStringArray<O>>());
        meta.add_member("buffer_data_", value_data)?;
        meta.add_member("buffer_offsets_", value_offsets)?;
        if let Some(null_bitmap) = null_bitmap {
            meta.add_member("null_bitmap_", null_bitmap)?;
        } else {
            meta.add_member("null_bitmap_", Blob::empty(client)?)?;
        }
        meta.add_usize("length_", self.length);
        meta.add_usize("offset_", self.offset);
        meta.add_usize("null_count_", self.null_count);
        meta.set_nbytes(nbytes);
        let metadata = client.create_metadata(&meta)?;
        return BaseStringArray::<O>::new_boxed(metadata);
    }
}

impl<O: OffsetSizeTrait> BaseStringBuilder<O> {
    pub fn new(client: &mut IPCClient, array: &GenericStringArray<O>) -> Result<Self> {
        use arrow_array::Array;

        let value_data = build_buffer(client, array.values())?;
        let value_offsets = build_offsets_buffer(client, array.offsets())?;
        let null_bitmap = build_null_bitmap_buffer(client, array.nulls())?;
        return Ok(BaseStringBuilder {
            sealed: false,
            length: array.len(),
            offset: 0,
            null_count: array.null_count(),
            value_data,
            value_offsets,
            null_bitmap,
            phantom: PhantomData,
        });
    }

    pub fn new_from_builder(
        client: &mut IPCClient,
        builder: &mut GenericStringBuilder<O>,
    ) -> Result<Self> {
        let array = builder.finish();
        return Self::new(client, &array);
    }

    pub fn len(&self) -> usize {
        return self.length;
    }

    pub fn is_empty(&self) -> bool {
        return self.length == 0;
    }

    pub fn offset(&self) -> usize {
        return self.offset;
    }

    pub fn null_count(&self) -> usize {
        return self.null_count;
    }

    pub fn as_slice(&mut self) -> &[u8] {
        return self.value_data.as_slice();
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        return unsafe { std::mem::transmute(self.value_data.as_mut_slice()) };
    }

    pub fn as_slice_offsets(&mut self) -> &[O] {
        return unsafe { std::mem::transmute(self.value_offsets.as_slice()) };
    }

    pub fn as_mut_slice_offsets(&mut self) -> &mut [O] {
        return unsafe { std::mem::transmute(self.value_offsets.as_mut_slice()) };
    }
}

pub fn downcast_to_array(object: Box<dyn Object>) -> Result<Box<dyn Array>> {
    macro_rules! downcast {
        ($object: ident, $ty: ty) => {
            |$object| match $object.downcast::<$ty>() {
                Ok(array) => Ok(array),
                Err(original) => Err(original),
            }
        };
    }

    let mut object: std::result::Result<Box<dyn Array>, Box<dyn Object>> = Err(object);
    object = object
        .or_else(downcast!(object, Int8Array))
        .or_else(downcast!(object, UInt8Array))
        .or_else(downcast!(object, Int16Array))
        .or_else(downcast!(object, UInt16Array))
        .or_else(downcast!(object, Int32Array))
        .or_else(downcast!(object, UInt32Array))
        .or_else(downcast!(object, Int64Array))
        .or_else(downcast!(object, UInt64Array))
        .or_else(downcast!(object, Float32Array))
        .or_else(downcast!(object, Float64Array))
        .or_else(downcast!(object, StringArray))
        .or_else(downcast!(object, LargeStringArray));

    match object {
        Ok(array) => return Ok(array),
        Err(object) => {
            return Err(VineyardError::invalid(format!(
                "downcast object to array failed, object type is: '{}'",
                object.meta().get_typename()?,
            )))
        }
    };
}

pub fn build_array(client: &mut IPCClient, array: ArrayRef) -> Result<Box<dyn Object>> {
    macro_rules! build {
        ($array: ident, $array_ty: ty, $builder_ty: ty) => {
            |$array| match $array.as_any().downcast_ref::<$array_ty>() {
                Some(array) => match <$builder_ty>::new(client, array) {
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
        .or_else(build!(array, array::Int8Array, Int8Builder))
        .or_else(build!(array, array::UInt8Array, UInt8Builder))
        .or_else(build!(array, array::Int16Array, Int16Builder))
        .or_else(build!(array, array::UInt16Array, UInt16Builder))
        .or_else(build!(array, array::Int32Array, Int32Builder))
        .or_else(build!(array, array::UInt32Array, UInt32Builder))
        .or_else(build!(array, array::Int64Array, Int64Builder))
        .or_else(build!(array, array::UInt64Array, UInt64Builder))
        .or_else(build!(array, array::Float32Array, Float32Builder))
        .or_else(build!(array, array::Float64Array, Float64Builder))
        .or_else(build!(array, array::StringArray, StringBuilder))
        .or_else(build!(array, array::LargeStringArray, LargeStringBuilder));

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

#[derive(Debug)]
pub struct SchemaProxy {
    meta: ObjectMeta,
    schema: schema::Schema,
}

impl TypeName for SchemaProxy {
    fn typename() -> &'static str {
        return staticize("vineyard::SchemaProxy");
    }
}

impl Default for SchemaProxy {
    fn default() -> Self {
        SchemaProxy {
            meta: ObjectMeta::default(),
            schema: schema::Schema::empty(),
        }
    }
}

impl Object for SchemaProxy {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        let schema: Vec<u8> = match self.meta.get_value("schema_binary_")? {
            Value::Object(values) => {
                let schema = values.get("bytes").ok_or(VineyardError::invalid(
                    "construct schema from binary failed: failed to get schema binary",
                ))?;
                match schema {
                    Value::Array(array) => {
                        let mut values = Vec::with_capacity(array.len());
                        for v in array {
                            match v {
                                Value::Number(n) => {
                                    if let Some(n) = n.as_u64() {
                                        values.push(n as u8);
                                    } else {
                                        return Err(VineyardError::invalid(
                                            format!("construct schema from binary failed: failed to get schema binary: not a positive number: {:?}", n),
                                        ));
                                    }
                                }
                                _ => return Err(VineyardError::invalid(
                                    format!("construct schema from binary failed: failed to get schema binary: not a positive number: {:?}", v),
                                )),
                            }
                        }
                        Ok(values)
                    }
                    _ => Err(VineyardError::invalid(
                        "construct schema from binary failed: value is not an array",
                    )),
                }
            }
            _ => Err(VineyardError::invalid(
                "construct schema from binary failed: failed to get schema binary",
            )),
        }?;
        self.schema = arrow_ipc::convert::try_schema_from_ipc_buffer(schema.as_slice())?;
        return Ok(());
    }
}

register_vineyard_object!(SchemaProxy);

impl SchemaProxy {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut schema = Box::<Self>::default();
        schema.construct(meta)?;
        return Ok(schema);
    }
}

impl AsRef<schema::Schema> for SchemaProxy {
    fn as_ref(&self) -> &schema::Schema {
        return &self.schema;
    }
}

pub struct SchemaProxyBuilder {
    sealed: bool,
    schema_binary: Vec<u8>,
    schema_textual: String,
}

impl ObjectBuilder for SchemaProxyBuilder {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl ObjectBase for SchemaProxyBuilder {
    fn build(&mut self, _client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<SchemaProxy>());
        meta.add_value(
            "schema_binary_",
            json!(
                {
                "bytes": self.schema_binary,
                }
            ),
        );
        meta.add_string("schema_textual_", self.schema_textual);
        meta.set_nbytes(self.schema_binary.len());
        let metadata = client.create_metadata(&meta)?;
        return SchemaProxy::new_boxed(metadata);
    }
}

impl SchemaProxyBuilder {
    pub fn new(_client: &mut IPCClient, schema: &schema::Schema) -> Result<Self> {
        let buffer: Vec<u8> = Vec::new();
        let writer = arrow_ipc::writer::StreamWriter::try_new(buffer, schema)?;
        let schema_binary = writer.into_inner()?;
        let schema_textual = schema.to_string();
        return Ok(SchemaProxyBuilder {
            sealed: false,
            schema_binary,
            schema_textual,
        });
    }

    pub fn new_from_builder(
        client: &mut IPCClient,
        builder: schema::SchemaBuilder,
    ) -> Result<Self> {
        return Self::new(client, &builder.finish());
    }
}

#[derive(Debug)]
pub struct RecordBatch {
    meta: ObjectMeta,
    batch: array::RecordBatch,
}

impl_typename!(RecordBatch, "vineyard::RecordBatch");

impl Default for RecordBatch {
    fn default() -> Self {
        RecordBatch {
            meta: ObjectMeta::default(),
            batch: array::RecordBatch::new_empty(Arc::new(schema::Schema::empty())),
        }
    }
}

impl Object for RecordBatch {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        let schema = self.meta.get_member::<SchemaProxy>("schema_")?;
        let schema = schema.as_ref().as_ref().clone();
        let _num_rows = self.meta.get_usize("row_num_")?;
        let _num_columns = self.meta.get_usize("column_num_")?;
        let columns_size = self.meta.get_usize("__columns_-size")?;
        let mut arrays = Vec::with_capacity(columns_size);
        for i in 0..columns_size {
            let column = self.meta.get_member_untyped(&format!("__columns_-{}", i))?;
            arrays.push(downcast_to_array(column)?.array());
        }
        self.batch = array::RecordBatch::try_new(Arc::new(schema), arrays)?;
        return Ok(());
    }
}

register_vineyard_object!(RecordBatch);

impl RecordBatch {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut batch = Box::<Self>::default();
        batch.construct(meta)?;
        return Ok(batch);
    }

    pub fn schema(&self) -> Arc<schema::Schema> {
        return self.batch.schema();
    }

    pub fn num_rows(&self) -> usize {
        return self.batch.num_rows();
    }

    pub fn num_columns(&self) -> usize {
        return self.batch.num_columns();
    }
}

impl AsRef<array::RecordBatch> for RecordBatch {
    fn as_ref(&self) -> &array::RecordBatch {
        return &self.batch;
    }
}

pub struct RecordBatchBuilder {
    sealed: bool,
    schema: SchemaProxyBuilder,
    row_num: usize,
    column_num: usize,
    columns: Vec<Box<dyn Object>>,
}

impl ObjectBuilder for RecordBatchBuilder {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl ObjectBase for RecordBatchBuilder {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        self.schema.build(client)?;
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<RecordBatch>());
        meta.add_member("schema_", self.schema.seal(client)?)?;
        meta.add_usize("row_num_", self.row_num);
        meta.add_usize("column_num_", self.column_num);
        meta.add_usize("__columns_-size", self.columns.len());
        for (i, column) in self.columns.into_iter().enumerate() {
            meta.add_member(&format!("__columns_-{}", i), column)?;
        }
        let metadata = client.create_metadata(&meta)?;
        return RecordBatch::new_boxed(metadata);
    }
}

impl RecordBatchBuilder {
    pub fn new(client: &mut IPCClient, batch: &array::RecordBatch) -> Result<Self> {
        let mut columns = Vec::with_capacity(batch.num_columns());
        for i in 0..batch.num_columns() {
            let array = batch.column(i);
            let array = build_array(client, array.clone())?;
            columns.push(array);
        }
        return Self::new_from_columns(
            client,
            batch.schema().as_ref(),
            batch.num_rows(),
            batch.num_columns(),
            columns,
        );
    }

    pub fn new_from_columns(
        client: &mut IPCClient,
        schema: &arrow_schema::Schema,
        row_num: usize,
        column_num: usize,
        columns: Vec<Box<dyn Object>>,
    ) -> Result<Self> {
        return Ok(RecordBatchBuilder {
            sealed: false,
            schema: SchemaProxyBuilder::new(client, schema)?,
            row_num: row_num,
            column_num: column_num,
            columns: columns,
        });
    }
}

#[derive(Debug)]
pub struct Table {
    meta: ObjectMeta,
    schema: schema::Schema,
    num_rows: usize,
    num_columns: usize,
    batches: Vec<Box<RecordBatch>>,
}

impl_typename!(Table, "vineyard::Table");

impl Default for Table {
    fn default() -> Self {
        Table {
            meta: ObjectMeta::default(),
            schema: schema::Schema::empty(),
            num_rows: 0,
            num_columns: 0,
            batches: Vec::new(),
        }
    }
}

impl Object for Table {
    fn construct(&mut self, meta: ObjectMeta) -> Result<()> {
        vineyard_assert_typename(typename::<Self>(), meta.get_typename()?)?;
        self.meta = meta;
        let schema = self.meta.get_member::<SchemaProxy>("schema_")?;
        let schema = schema.as_ref().as_ref().clone();
        self.num_rows = self.meta.get_usize("num_rows_")?;
        self.num_columns = self.meta.get_usize("num_columns_")?;
        let _batch_num = self.meta.get_usize("batch_num_")?;
        let partitions_size = self.meta.get_usize("partitions_-size")?;
        let mut batches = Vec::with_capacity(partitions_size);
        for i in 0..partitions_size {
            let batch = self
                .meta
                .get_member::<RecordBatch>(&format!("partitions_-{}", i))?;
            batches.push(batch);
        }
        self.schema = schema;
        self.batches = batches;
        return Ok(());
    }
}

register_vineyard_object!(Table);

impl Table {
    pub fn new_boxed(meta: ObjectMeta) -> Result<Box<dyn Object>> {
        let mut table = Box::<Self>::default();
        table.construct(meta)?;
        return Ok(table);
    }

    pub fn schema(&self) -> &schema::Schema {
        return &self.schema;
    }

    pub fn num_rows(&self) -> usize {
        return self.num_rows;
    }

    pub fn num_columns(&self) -> usize {
        return self.num_columns;
    }

    pub fn num_batches(&self) -> usize {
        return self.batches.len();
    }

    pub fn batches(&self) -> &[Box<RecordBatch>] {
        return &self.batches;
    }
}

impl AsRef<[Box<RecordBatch>]> for Table {
    fn as_ref(&self) -> &[Box<RecordBatch>] {
        return &self.batches;
    }
}

pub struct TableBuilder {
    sealed: bool,
    global: bool,
    schema: SchemaProxyBuilder,
    num_rows: usize,
    num_columns: usize,
    batches: Vec<Box<dyn Object>>,
}

impl ObjectBuilder for TableBuilder {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl ObjectBase for TableBuilder {
    fn build(&mut self, client: &mut IPCClient) -> Result<()> {
        if self.sealed {
            return Ok(());
        }
        self.set_sealed(true);
        self.schema.build(client)?;
        return Ok(());
    }

    fn seal(mut self, client: &mut IPCClient) -> Result<Box<dyn Object>> {
        self.build(client)?;
        let mut meta = ObjectMeta::new_from_typename(typename::<Table>());
        meta.set_global(self.global);
        meta.add_member("schema_", self.schema.seal(client)?)?;
        meta.add_usize("num_rows_", self.num_rows);
        meta.add_usize("num_columns_", self.num_columns);
        meta.add_usize("batch_num_", self.batches.len());
        meta.add_usize("partitions_-size", self.batches.len());
        for (i, batch) in self.batches.into_iter().enumerate() {
            meta.add_member(&format!("partitions_-{}", i), batch)?;
        }
        let metadata = client.create_metadata(&meta)?;
        return Table::new_boxed(metadata);
    }
}

impl TableBuilder {
    pub fn new(
        client: &mut IPCClient,
        schema: &schema::Schema,
        table: &[array::RecordBatch],
    ) -> Result<Self> {
        let schema = SchemaProxyBuilder::new(client, schema)?;

        let mut batches = Vec::with_capacity(table.len());
        let mut num_rows = 0;
        let mut num_columns = 0;
        for batch in table {
            num_rows += batch.num_rows();
            num_columns = batch.num_columns();
            let batch = RecordBatchBuilder::new(client, batch)?;
            batches.push(batch.seal(client)?);
        }
        return Ok(TableBuilder {
            sealed: false,
            global: false,
            schema: schema,
            num_rows: num_rows,
            num_columns: num_columns,
            batches: batches,
        });
    }

    pub fn new_from_batches(
        client: &mut IPCClient,
        schema: &schema::Schema,
        num_rows: usize,
        num_columns: usize,
        batches: Vec<Box<dyn Object>>,
    ) -> Result<Self> {
        return Ok(TableBuilder {
            sealed: false,
            global: false,
            schema: SchemaProxyBuilder::new(client, schema)?,
            num_rows: num_rows,
            num_columns: num_columns,
            batches: batches,
        });
    }

    pub fn new_from_batch_columns(
        client: &mut IPCClient,
        schema: &schema::Schema,
        num_rows: Vec<usize>,
        num_columns: usize,
        batches: Vec<Vec<Box<dyn Object>>>,
    ) -> Result<Self> {
        let mut chunks = Vec::with_capacity(batches.len());
        let mut total_num_rows = 0;
        for (num_row, batch) in izip!(num_rows, batches) {
            total_num_rows += num_row;
            let batch =
                RecordBatchBuilder::new_from_columns(client, schema, num_row, num_columns, batch)?;
            chunks.push(batch.seal(client)?);
        }
        return Ok(TableBuilder {
            sealed: false,
            global: false,
            schema: SchemaProxyBuilder::new(client, schema)?,
            num_rows: total_num_rows,
            num_columns: num_columns,
            batches: chunks,
        });
    }
}

pub(crate) fn resolve_buffer(meta: &ObjectMeta, key: &str) -> Result<Buffer> {
    let id = meta.get_member_id(key)?;
    match meta.get_buffer(id)? {
        None => {
            return Err(VineyardError::invalid(format!(
                "buffer '{}' not exists in metadata",
                key
            )));
        }
        Some(buffer) => {
            return Ok(buffer);
        }
    }
}

pub(crate) fn resolve_null_bitmap_buffer(
    meta: &ObjectMeta,
    key: &str,
) -> Result<Option<NullBuffer>> {
    let id = meta.get_member_id(key)?;
    if is_blob(id) {
        return Ok(None);
    }
    if let Ok(buffer) = resolve_buffer(meta, key) {
        let length = meta.get_usize("length_")?;
        let null_count = meta.get_usize("null_count_")?;
        let offset = meta.get_usize("offset_")?;
        let buffer = BooleanBuffer::new(buffer, offset, length);
        return Ok(Some(unsafe {
            NullBuffer::new_unchecked(buffer, null_count)
        }));
    }
    return Ok(None);
}

pub(crate) fn resolve_scalar_buffer<T: NumericType>(
    meta: &ObjectMeta,
    key: &str,
) -> Result<TypedBuffer<T>> {
    let buffer = resolve_buffer(meta, key)?;
    let length = meta
        .get_usize("length_")
        .unwrap_or(buffer.len() / std::mem::size_of::<T>());
    let offset = meta.get_usize("offset_").unwrap_or(0);
    return Ok(TypedBuffer::<T>::new(buffer, offset, length));
}

pub(crate) fn resolve_offsets_buffer<O: OffsetSizeTrait>(
    meta: &ObjectMeta,
    key: &str,
) -> Result<OffsetBuffer<O>> {
    let buffer = resolve_buffer(meta, key)?;
    let length = meta.get_usize("length_")? + 1;
    let offset = meta.get_usize("offset_")?;
    let buffer = ScalarBuffer::<O>::new(buffer, offset, length);
    return Ok(unsafe { OffsetBuffer::new_unchecked(buffer) });
}

pub(crate) fn build_buffer(client: &mut IPCClient, buffer: &Buffer) -> Result<BlobWriter> {
    let mut blob = client.create_blob(buffer.len())?;
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.as_ptr(), blob.as_typed_mut_ptr::<u8>(), buffer.len());
    };
    return Ok(blob);
}

pub(crate) fn build_null_bitmap_buffer(
    client: &mut IPCClient,
    buffer: Option<&NullBuffer>,
) -> Result<Option<BlobWriter>> {
    match buffer {
        None => {
            return Ok(None);
        }
        Some(buffer) => {
            let null_bitmap = build_buffer(client, buffer.buffer())?;
            return Ok(Some(null_bitmap));
        }
    }
}

pub(crate) fn build_scalar_buffer<T: NumericType>(
    client: &mut IPCClient,
    buffer: &TypedBuffer<T>,
) -> Result<BlobWriter> {
    let values = build_buffer(client, buffer.inner())?;
    return Ok(values);
}

pub(crate) fn build_offsets_buffer<O: OffsetSizeTrait>(
    client: &mut IPCClient,
    buffer: &OffsetBuffer<O>,
) -> Result<BlobWriter> {
    let offsets = build_buffer(client, buffer.inner().inner())?;
    return Ok(offsets);
}
