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
use super::Create;
use super::IPCClient;
use super::{ObjectBase, Object, ObjectBuilder, Registered};
use super::ObjectMeta;


#[derive(Debug, Clone)]
pub struct Array<T> {
    meta: ObjectMeta,
    id: ObjectID,
    registered: bool,
    size: usize,
    buffer: Arc<Mutex<Blob>>, // Question: I changed Rc into Arc for Send trait
    phantom: PhantomData<T>, // Question: if this is correct?
}

impl<T> Create for Array<T> {
    fn create() -> &'static Arc<Mutex<Box<dyn Object>>> {
        lazy_static! {
            static ref SHARED_ARRAY: Arc<Mutex<Box<dyn Object>>> =
                Arc::new(Mutex::new(Box::new(Array::default() as Array<i32>))); // Question
        }
        &SHARED_ARRAY
    }
}

impl<T> Default for Array<T> {
    fn default() -> Array<T> {
        Array{
            meta: ObjectMeta::default(),
            id: invalid_object_id(),
            registered: false,
            size: 0,
            buffer: Arc::new(Mutex::new(Blob::default())), // Question: I changed Rc into Arc for Send trait
            phantom: PhantomData,
        }
    }
}

impl<T> Array<T> {

    pub fn construct(&mut self, meta: &ObjectMeta) {
        let __type_name: String = type_name::<Array<T>>().to_string();
        CHECK(meta.get_type_name() == __type_name);
        self.meta = meta.clone();
        self.id = meta.get_id();
        self.size = meta.get_key_value(&"size_".to_string()).as_u64().unwrap() as usize;
        //self.buffer = meta.get_member(&"buffer_".to_string()); // Question: Rc or Arc<Mutex>
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
        self.buffer.lock().unwrap().data()
    }
}

impl<T: Send + Clone> Registered for Array<T> {}

impl<T: Send + Clone> Object for Array<T> {
    fn meta(&self) -> &ObjectMeta {
        &self.meta
    }


    fn meta_mut(&mut self) -> &mut ObjectMeta{
        &mut self.meta
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


