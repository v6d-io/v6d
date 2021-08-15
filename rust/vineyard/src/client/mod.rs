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

#[allow(clippy::module_inception)]
pub mod client;
pub mod ds;
pub mod ipc_client;
pub mod rpc_client;

pub use self::ds::blob::Blob;
pub use self::ds::blob::BlobWriter;

pub use self::ds::object_factory::ObjectFactory;

pub use self::ds::object_meta::ObjectMeta;

pub use self::ds::object::Object;

pub use crate::common::util::InstanceID;
pub use crate::common::util::ObjectID;
