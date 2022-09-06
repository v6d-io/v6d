use std::io;

pub fn VINEYARD_CHECK_OK<T>(status: io::Result<T>) {
    if let Err(_) = status {
        panic!("Error occurs.")
    }
}

pub fn VINEYARD_ASSERT(condition: bool) {
    if !condition {
        panic!()
    }
}

pub fn RETURN_ON_ASSERT(b: bool) {
    if !b {
        panic!("On assert failed.");
    }
}

pub fn RETURN_ON_ERROR<T>(status: io::Result<T>) {
    if let Err(_) = status {
        panic!("Error occurs.")
    }
}

// Question
pub fn CHECK(condition: bool) {
    if !condition {
        panic!()
    }
}
