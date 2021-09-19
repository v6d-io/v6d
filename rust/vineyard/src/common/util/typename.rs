use std::io;

pub fn type_name<T>() -> &'static str {
    return std::any::type_name::<T>();
}