pub type ObjectID = u64;
pub type InstanceID = u64;
pub type Signature = u64;

// TODO: Rust parse check
pub fn object_id_from_string(s: &String) -> ObjectID {
    s.parse::<ObjectID>().unwrap()
}

pub fn object_id_to_string(id: ObjectID) -> String {
    String::new()
}

pub fn is_blob(id: ObjectID) -> bool {
    false
}
