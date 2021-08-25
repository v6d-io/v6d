pub fn VINEYARD_CHECK_OK(status: bool) {
    
}


pub fn VINEYARD_ASSERT(condition: bool) {
    if !condition {
        panic!()
    }
}

pub fn RETURN_ON_ASSERT(b: bool) {
    if !b {
        panic!()
    }
}


