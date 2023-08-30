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

#![allow(clippy::box_default)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_return)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::redundant_field_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::vec_box)]
#![allow(incomplete_features)]
#![allow(non_upper_case_globals)]
#![cfg_attr(feature = "nightly", feature(associated_type_defaults))]
#![cfg_attr(feature = "nightly", feature(box_into_inner))]
#![cfg_attr(feature = "nightly", feature(specialization))]
#![cfg_attr(feature = "nightly", feature(trait_alias))]
#![cfg_attr(feature = "nightly", feature(unix_socket_peek))]

#[macro_use]
extern crate log;

#[macro_use]
extern crate lazy_static;

pub mod client;
pub mod common;
pub mod ds;

pub use common::util::status::{Result, VineyardError};
