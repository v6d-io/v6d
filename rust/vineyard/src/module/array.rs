use std::io;
use std::mem;

use super::status::*;
use super::uuid::*;
use super::BlobWriter;
use super::IPCClient;

#[derive(Debug)]
pub struct ArrayBuilder<T> {
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
