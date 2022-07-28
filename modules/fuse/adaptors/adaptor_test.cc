#include<iostream>
#include <fstream>

#include<memory>
#include<string>
#include"factory/deserializer_registry.h"

int main(){
    vineyard::fuse::register_once();


    std::string ipc_socket = "/var/run/vineyard.sock";
    vineyard::logging::InitGoogleLogging("vineyard");
    vineyard::Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
  
    // vineyard::ObjectID bool_id = 0x0000044b814c2f84;
    vineyard::ObjectID int_id = 0x00020e02666c8964;
    // vineyard::ObjectID string_id = 0x00000f00815840fc;
    auto r =client.GetObject(int_id);
    // auto r_bool = std::dynamic_pointer_cast<vineyard::BooleanArray>(client.GetObject(bool_id));
    // vineyard::ObjectMeta  bool_meta;
    // VINEYARD_CHECK_OK(client.GetMetaData(bool_id,bool_meta));
    // VLOG(0)<< bool_meta.GetTypeName();
    // auto r_string =  std::dynamic_pointer_cast<vineyard::NumericArray<std::string>>(client.GetObject(string_id));
    // VLOG(0) << "get int array";
    // auto int_array = r_int->GetArray();
    // VLOG(0)<<"got int array";

    // VLOG(0)<<"get string array";
    // auto string_array = r_string->GetArray();
    // VLOG(0)<<"got string array";

    // VLOG(0)<<"get bool array";
    
    // auto bool_array = r_bool->GetArray();
    // VLOG(0)<<"got bool array";
    // for (int64_t i = 0; i < bool_array->length(); ++i) {
    //  std::cout<<bool_array->Value(i)<<std::endl;
    // }
  // // VLOG(0) << "this is level 0";
  // // VLOG(2) << "this is level 2";
  // // VLOG(4) << "this is level 4"

    std::string tn = r->meta().GetTypeName();
    // std::string tn = type_name<vineyard::NumericArray<int64_t>>();
    std::clog<<tn<<std::endl;
    
    auto p = vineyard::fuse::d_array_registry.at(tn);
    auto b = p(r);

    std::ofstream myfile;
    myfile.open ("example.txt",std::ios::trunc|std::ios::out	);
    myfile << b->ToString();
    myfile.close();
    return 0;
}