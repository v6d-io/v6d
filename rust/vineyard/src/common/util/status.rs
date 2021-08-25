
pub fn RETURN_ON_ASSERT(b: bool) {
    if !b {
        panic!()
    }
}