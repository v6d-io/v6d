use std::io;
use std::marker::PhantomData;
use std::mem;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use lazy_static::lazy_static;

use super::status::*;
use super::typename::type_name;
use super::uuid::*;
use super::Create;
use super::IPCClient;
use super::ObjectMeta;
use super::{Blob, BlobWriter};
use super::{Object, ObjectBase, ObjectBuilder, Registered};

#[derive(Debug, Clone)]
pub struct Array<T> {
    meta: ObjectMeta,
    id: ObjectID,
    registered: bool,
    size: usize,
    buffer: Rc<Blob>, // Question: unsafe Send
    phantom: PhantomData<T>,  // Question: if this is correct?
}

unsafe impl<T> Send for Array<T> {}

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
        Array {
            meta: ObjectMeta::default(),
            id: invalid_object_id(),
            registered: false,
            size: 0,
            buffer: Rc::new(Blob::default()), 
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
        //self.buffer = meta.get_member(&"buffer_".to_string()); 
        // Question: Rust do not support dynamic_pointer_cast; 
        // how to ensure it returns a Blob
    }

    pub fn operator(&self, loc: isize) -> *const u8 {
        unsafe { self.data().offset(loc) }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn data(&self) -> *const u8 {
        self.buffer.data()
    }
}

impl<T: Send + Clone> Registered for Array<T> {}

impl<T: Send + Clone> Object for Array<T> {
    fn meta(&self) -> &ObjectMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ObjectMeta {
        &mut self.meta
    }

    fn id(&self) -> ObjectID {
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



pub trait ArrayBaseBuilder: ObjectBuilder {}



pub struct ArrayBuilder<T> {
    buffer: Rc<dyn ObjectBase>,
    buffer_writer: Box<BlobWriter>,
    data: T,
    size: usize,
    sealed: bool,
}

impl<T> ArrayBaseBuilder for ArrayBuilder<T> {}

impl<T> ObjectBuilder for ArrayBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }
}

impl<T> ObjectBase for ArrayBuilder<T> {}

impl<T> ArrayBuilder<T> {
    // pub fn from(client: &impl Client, size: usize) -> ArrayBuilder<T> {
    //     VINEYARD_CHECK_OK(client.create_blob(size * mem::size_of<T>(), buffer_writer));
    //     ArrayBuilder{
    //         size: size,
    //         data: buffer_writer.data() //TODO
    //     }
    // }
}

pub struct ResizableArrayBuilder<T> {
    size: usize,
    buffer: Rc<dyn ObjectBase>,
    vec: Vec<T>,
    sealed: bool,
}


impl<T> ArrayBaseBuilder for ResizableArrayBuilder<T> {}

impl<T> ObjectBuilder for ResizableArrayBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }
}

impl<T> ObjectBase for ResizableArrayBuilder<T> {}