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

use super::{Client, IPCClient};


pub struct ObjectMeta {
    pub client: Option<Box<dyn Client>>, 
    pub meta: String, // TODO: json

}

impl ObjectMeta{
    
    fn new() -> ObjectMeta{
        ObjectMeta{
            client: None,
            meta: String::new(),
        }
    }

    fn set_client(&mut self, client: Box<dyn Client>) {
        self.client = Some(client);
    }

    fn get_client(&self) -> &Box<dyn Client>{
        match &self.client{
            Some(client)=> &client,
            None => panic!("The object has no client"),
        }
    }

    // fn set_id() {}

    // fn get_id() {}

    // fn get_signature() {}

    // fn set_global() {}

    // fn is_global() {}

    // fn set_type_name() {}

    // fn get_type_name() {}  
    
    // fn set_n_bytes() {}

    // fn get_n_bytes() {}

    // fn get_instance_id() {}

    // fn is_local() {}

    // fn has_key() {}

    // fn reset_key() {}

    // fn add_key_value() {}

    // fn get_key_value() {}

    // fn add_member() {}

    // fn get_member() {}

    // fn get_member_meta() {}

    // fn get_buffer() {}

    // fn set_buffer() {}

    
    


}

