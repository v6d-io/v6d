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

use super::ObjectMeta;


pub trait Client {
    fn connect(&self, socket: &str) -> bool;
    
    fn disconnect(&self);

    fn connected(&self) -> bool;

    fn get_meta_data(&self, object_id: u64, sync_remote: bool) -> ObjectMeta;
}


