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

use arrow_array::array::*;
use arrow_array::builder::*;
use arrow_array::types::*;
use arrow_schema::DataType;

pub trait ToArrowType {
    type Type;
    type ArrayType;
    type BuilderType;

    fn datatype() -> DataType;
}

macro_rules! impl_to_arrow_type {
    ($ty:ty, $datatype:expr, $type:ty, $array_ty:ty, $builder_ty:ty) => {
        impl ToArrowType for $ty {
            type Type = $type;
            type ArrayType = $array_ty;
            type BuilderType = $builder_ty;

            fn datatype() -> DataType {
                return $datatype;
            }
        }
    };
}

impl_to_arrow_type!(i8, DataType::Int8, Int8Type, Int8Array, Int8Builder);
impl_to_arrow_type!(u8, DataType::UInt8, UInt8Type, UInt8Array, UInt8Builder);
impl_to_arrow_type!(i16, DataType::Int16, Int16Type, Int16Array, Int16Builder);
impl_to_arrow_type!(
    u16,
    DataType::UInt16,
    UInt16Type,
    UInt16Array,
    UInt16Builder
);
impl_to_arrow_type!(i32, DataType::Int32, Int32Type, Int32Array, Int32Builder);
impl_to_arrow_type!(
    u32,
    DataType::UInt32,
    UInt32Type,
    UInt32Array,
    UInt32Builder
);
impl_to_arrow_type!(i64, DataType::Int64, Int64Type, Int64Array, Int64Builder);
impl_to_arrow_type!(
    u64,
    DataType::UInt64,
    UInt64Type,
    UInt64Array,
    UInt64Builder
);
impl_to_arrow_type!(
    f32,
    DataType::Float32,
    Float32Type,
    Float32Array,
    Float32Builder
);
impl_to_arrow_type!(
    f64,
    DataType::Float64,
    Float64Type,
    Float64Array,
    Float64Builder
);
