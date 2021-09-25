use std::any::Any;
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
use super::ENSURE_NOT_SEALED;
use super::{Blob, BlobWriter};
use super::{Object, ObjectBase, ObjectBuilder, Registered};

#[derive(Debug, Clone)]
pub struct Array<T> {
    meta: ObjectMeta,
    id: ObjectID,
    registered: bool,
    size: usize,
    buffer: Rc<Blob>,        // Question: unsafe Send // 不行用Arc
    phantom: PhantomData<T>, 
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
        let member: &dyn Any = &meta.get_member(&"buffer_".to_string());
        match member.downcast_ref::<Blob>() {
            // 去掉ref
            Some(blob) => self.buffer = Rc::new((*blob).clone()),
            None => panic!("The member isn't a Blob."),
        };
        //self.buffer = meta.get_member(&"buffer_".to_string());
        // Question: how to ensure it returns a Blob; Use blob.clone() after downcasting
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

impl<T: Send + Clone + std::fmt::Debug> Registered for Array<T> {}

impl<T: Send + Clone + std::fmt::Debug> Object for Array<T> {
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

pub trait ArrayBaseBuilder: ObjectBuilder {
    fn from(&mut self, client: &IPCClient) {}

    fn from_array<T>(&mut self, value: &Array<T>) {
        self.set_size(value.size);
        let buf = Rc::clone(&value.buffer);
        let buf: Rc<dyn ObjectBase> = buf;
        self.set_buffer(&buf);
    }

    fn from_shared_array<T>(&mut self, value: &Rc<Array<T>>) {
        self.from_array(&**value)
    }

    fn seal<T>(&mut self, client: &IPCClient) -> Rc<dyn Object>
    where
        Self: Sized,
    {
        ENSURE_NOT_SEALED(self);
        let mut value: Array<T> = Array::default();
        let value_nbytes: usize = 0;
        value
            .meta
            .set_type_name(&type_name::<Array<T>>().to_string());
        // if (std::is_base_of<GlobalObject, Array<T>>::value)
        panic!();
    }

    fn set_size(&mut self, size: usize);
    fn set_buffer(&mut self, buffer: &Rc<dyn ObjectBase>);
}

pub struct ArrayBuilder<T> {
    buffer: Rc<dyn ObjectBase>,
    buffer_writer: Box<BlobWriter>,
    data: T,
    size: usize,
    sealed: bool,
}

impl<T> ArrayBaseBuilder for ArrayBuilder<T> {
    fn set_size(&mut self, size: usize) {
        self.size = size;
    }

    fn set_buffer(&mut self, buffer: &Rc<dyn ObjectBase>) {
        self.buffer = Rc::clone(buffer);
    }
}

impl<T> ObjectBuilder for ArrayBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
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

impl<T> ArrayBaseBuilder for ResizableArrayBuilder<T> {
    fn set_size(&mut self, size: usize) {
        self.size = size;
    }

    fn set_buffer(&mut self, buffer: &Rc<dyn ObjectBase>) {
        self.buffer = Rc::clone(buffer);
    }
}

impl<T> ObjectBuilder for ResizableArrayBuilder<T> {
    fn sealed(&self) -> bool {
        self.sealed
    }

    fn set_sealed(&mut self, sealed: bool) {
        self.sealed = sealed;
    }
}

impl<T> ObjectBase for ResizableArrayBuilder<T> {}
