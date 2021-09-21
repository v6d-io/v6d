use std::io;
use std::mem;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

use lazy_static::lazy_static;

use super::typename::type_name;
use super::status::*;
use super::uuid::*;
use super::{Blob, BlobWriter};
use super::IPCClient;
use super::{ObjectBase, Object, ObjectBuilder, Registered};
use super::ObjectMeta;


#[derive(Debug)]
pub struct Array<T> {
    pub meta: ObjectMeta,
    pub id: ObjectID,
    registered: bool,
    size: usize,
    buffer: Arc<Mutex<Blob>>, // Question: I change Rc into Arc for concurrency
    phantom: PhantomData<T>, // Question: if this is correct?
}

impl<T: Send> Array<T> {
    // pub fn create() -> &'static Arc<Mutex<dyn Object>> {
    //     lazy_static! {
    //         static ref SHARED_ARRAY: Arc<Mutex<dyn Object>> =
    //             Arc::new(Mutex::new(Object::default())); // Question: cast Array<T> to Object
    //     }
    //     &SHARED_ARRAY
    // }

    pub fn construct(&mut self, meta: &ObjectMeta) {
        let __type_name: String = type_name::<Array<T>>().to_string();
        CHECK(meta.get_type_name() == __type_name);
        self.meta = meta.clone();
        self.id = meta.get_id();
        self.size = meta.get_key_value(&"size_".to_string()).as_u64().unwrap() as usize;
        //self.buffer = meta.get_member(&"buffer_".to_string()); 
        // Question: cast Rc<Object> to Rc<Blob>
        // std::dynamic_pointer_cast<Blob>(meta.GetMember("buffer_"));
    }

    pub fn operator(&self, loc: isize) -> *const u8 {
        unsafe{
            self.data().offset(loc)
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn data(&self) -> *const u8 {
        self.buffer.data()
    }
}

impl<T: Send> Registered for Array<T> {}

impl<T: Send> Object for Array<T> {
    fn meta(&self) -> &ObjectMeta {
        &self.meta
    }

    fn id(&self) -> ObjectID{
        self.id
    }

    fn set_id(&mut self, id: ObjectID) {
        self.id = id;
    }

    fn set_meta(&mut self, meta: &ObjectMeta) {
        self.meta = meta.clone();
    }
}

impl<T: Send> ObjectBase for Array<T> {}

impl<T: Send> Default for Array<T> {
    fn default () -> Self {
        Array {
            meta: ObjectMeta::default(),
            id: 0,
            registered: false,
            size: 0,
            buffer: Rc::new(Blob::default()),
            phantom: PhantomData,
        }
    }
}

pub trait ArrayBaseBuilder {}

pub struct ArrayBuilder<T> {
    buffer: Rc<dyn ObjectBase>,
    buffer_writer: Box<BlobWriter>, 
    data: T,
    size: usize,
}


// impl<T> ArrayBuilder<T> {
//     pub fn create(client: &impl Client, size: usize) -> ArrayBuilder<T> {
//         VINEYARD_CHECK_OK(client.create_blob(size * mem::size_of<T>(), buffer_writer));
//         ArrayBuilder{
//             size: size,
//             data: buffer_writer.data() //TODO
//         }
//     }
// }

pub struct ResizableArrayBuilder<T> {
    size: usize,
    buffer: Rc<dyn ObjectBase>,
    vec: Vec<T>,
}


