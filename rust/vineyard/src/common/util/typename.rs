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

use static_str_ops::*;

/// A trait to generate specialized type name for given Rust type.
///
/// Note that the `typename()` method doesn't return `&'static str` for
/// optimization due the complain "use of generic parameter from outer
/// function" when we try to compute typename of `Array<T>` using the
/// `formatcp!` from the `const_format` crate.
///
/// We temporarily use `String` as the return type and leave it as a TODO.
pub trait TypeName {
    fn typename() -> &'static str
    where
        Self: Sized,
    {
        return staticize_once!(std::any::type_name::<Self>());
    }
}

/// Generate typename for given type in Rust.
pub fn typename<T: TypeName>() -> &'static str {
    return T::typename();
}

#[macro_export]
macro_rules! impl_typename {
    ($t:ty, $name:expr) => {
        impl TypeName for $t {
            fn typename() -> &'static str {
                return $name;
            }
        }
    };
}

pub use impl_typename;

impl_typename!(i8, "int8");
impl_typename!(u8, "uint8");
impl_typename!(i16, "int16");
impl_typename!(u16, "uint16");
impl_typename!(i32, "int");
impl_typename!(u32, "uint");
impl_typename!(i64, "int64");
impl_typename!(u64, "uint64");
impl_typename!(f32, "float");
impl_typename!(f64, "double");
impl_typename!(bool, "bool");
impl_typename!(String, "std::string");
