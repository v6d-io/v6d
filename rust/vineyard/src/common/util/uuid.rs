use rand::Rng;

pub type ObjectID = u64;
pub type InstanceID = u64;
pub type Signature = u64;

// pub fn get_blob_addr() {}

// pub fn generate_blob_id() {}

pub fn empty_blob_id() -> ObjectID {
    0x8000000000000000u64
}

pub fn generate_object_id() -> ObjectID {
    let mut rng = rand::thread_rng();
    0x7FFFFFFFFFFFFFFFu64 & rng.gen::<u64>()
}

pub fn generate_signature() -> Signature {
    let mut rng = rand::thread_rng();
    0x7FFFFFFFFFFFFFFFu64 & rng.gen::<u64>()
}

pub fn is_blob(id: ObjectID) -> bool {
    if id & 0x8000000000000000u64 == 0 {
        return false;
    }
    true
}

pub fn object_id_from_string(s: &String) -> ObjectID {
    s[1..].parse::<ObjectID>().unwrap()
}

pub fn object_id_to_string(id: ObjectID) -> String {
    format!("o{:x}", id)
}

pub fn signature_to_string(id: ObjectID) -> String {
    format!("s{:x}", id)
}

pub fn invalid_object_id() -> ObjectID {
    ObjectID::MAX
}

pub fn invalid_signature() -> ObjectID {
    Signature::MAX
}

pub fn unspecified_instance_id() -> InstanceID {
    InstanceID::MAX
}
