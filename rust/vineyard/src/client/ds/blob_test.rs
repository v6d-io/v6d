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

#[cfg(test)]
mod tests {
    use std::mem::ManuallyDrop;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use spectral::prelude::*;

    use super::super::super::*;
    use super::super::*;

    #[test]
    fn test_manually_drop() {
        static mut drop_a_called: AtomicUsize = AtomicUsize::new(0);
        static mut drop_b_called: AtomicUsize = AtomicUsize::new(0);

        struct A {}

        impl Drop for A {
            fn drop(&mut self) {
                // record a's dtor
                unsafe {
                    drop_a_called.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        impl A {
            pub fn tell(&self) {}
        }

        struct B {
            a: ManuallyDrop<A>,
        }

        impl Drop for B {
            fn drop(&mut self) {
                // a should live
                assert!(unsafe { drop_a_called.load(Ordering::SeqCst) } == 0);
                // record b's dtor
                unsafe {
                    drop_b_called.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        impl B {
            pub fn release(mut self) -> A {
                return unsafe { ManuallyDrop::take(&mut self.a) };
            }
        }

        let b = B {
            a: ManuallyDrop::new(A {}),
        };
        assert!(unsafe { drop_a_called.load(Ordering::SeqCst) } == 0);
        assert!(unsafe { drop_b_called.load(Ordering::SeqCst) } == 0);
        let a = b.release();
        assert!(unsafe { drop_a_called.load(Ordering::SeqCst) } == 0);
        assert!(unsafe { drop_b_called.load(Ordering::SeqCst) } == 1);
        a.tell();
        drop(a);
        assert!(unsafe { drop_a_called.load(Ordering::SeqCst) } == 1);
        assert!(unsafe { drop_b_called.load(Ordering::SeqCst) } == 1);
    }

    #[test]
    fn test_blob() -> Result<()> {
        const N: usize = 1024;

        let mut client = IPCClient::default()?;

        let mut blob_writer = client.create_blob(N)?;
        let blob_writer_id = blob_writer.id();
        assert_that!(blob_writer_id).is_greater_than(0);

        let slice_mut = blob_writer.as_mut_slice();
        for (idx, item) in slice_mut.iter_mut().enumerate() {
            *item = idx as u8;
        }

        // test seal
        {
            let object = blob_writer.seal(&mut client)?;
            let blob = downcast_object::<Blob>(object)?;
            let blob_id = blob.id();
            assert_that!(blob_id).is_greater_than(0);
            assert_that!(blob_id).is_equal_to(blob_writer_id);

            let slice = blob.as_slice()?;
            for (idx, item) in slice.iter().enumerate() {
                assert_that!(*item).is_equal_to(idx as u8);
            }
        }

        // test get blob
        {
            let blob = client.get_blob(blob_writer_id)?;
            let blob_id = blob.id();
            assert_that!(blob_id).is_greater_than(0);
            assert_that!(blob_id).is_equal_to(blob_writer_id);

            let slice = blob.as_slice()?;
            for (idx, item) in slice.iter().enumerate() {
                assert_that!(*item).is_equal_to(idx as u8);
            }
        }

        return Ok(());
    }
}
