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

use std::ptr::NonNull;
use std::sync::Arc;

use arrow_buffer::{alloc, Buffer, MutableBuffer};

/// An `arrow::alloc::Allocation` implementation to prevent the pointer
/// been freed by `arrow::Buffer`.
///
/// We use `arrow::Buffer` for both mutable and immutable cases as there's
/// no way to construct `arrow::MutableBuffer` from external allocated
/// memory without transferring the ownership, and we don't want the resize
/// interfaces on `arrow::MutableBuffer`.
///
/// Instead, we cast pointers (`*const u8` and `*mut u8`) to the expected type.
pub struct MmapAllocation {}
impl alloc::Allocation for MmapAllocation {}

lazy_static! {
    static ref MMAP_ALLOCATION: Arc<MmapAllocation> = Arc::new(MmapAllocation {});
}

pub fn arrow_buffer(pointer: *const u8, len: usize) -> Buffer {
    return arrow_buffer_mut(pointer as *mut u8, len);
}

pub fn arrow_buffer_with_offset(pointer: *const u8, offset: isize, len: usize) -> Buffer {
    return arrow_buffer_with_offset_mut(pointer as *mut u8, offset, len);
}

pub fn arrow_buffer_mut(pointer: *mut u8, len: usize) -> Buffer {
    return arrow_buffer_with_offset_mut(pointer as *mut u8, 0, len);
}

pub fn arrow_buffer_with_offset_mut(pointer: *mut u8, offset: isize, len: usize) -> Buffer {
    if pointer.is_null() {
        return MutableBuffer::new(0).into();
    } else {
        return unsafe {
            Buffer::from_custom_allocation(
                NonNull::new_unchecked(pointer.offset(offset)),
                len,
                MMAP_ALLOCATION.clone(),
            )
        };
    }
}

pub fn arrow_buffer_null() -> Buffer {
    return arrow_buffer_mut(std::ptr::null_mut(), 0);
}
