#include<iostream>
#include<memory>
#include<string>
#include"factory/deserializer_registry.h"

int main(){

    std::string tn = type_name<vineyard::NumericArray<int64_t>>();
    std::clog<<tn<<std::endl;
    vineyard::fuse::register_once();
    auto p = vineyard::fuse::d_registry.at(tn);
    p(nullptr);
    return 0;
}