#include "common/util/logging.h"
#include "client/client.h"
#include "basic/ds/arrow.h"
#include "common/util/typename.h"

#include <iostream>
typedef struct{
  using inttype = int;
} T;
int main() {
   std::string ipc_socket = "/var/run/vineyard.sock";
  vineyard::logging::InitGoogleLogging("vineyard");
  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
  
    vineyard::ObjectID bool_id = 0x0000044b814c2f84;
    vineyard::ObjectID int_id = 0x000009922241b85e;
    // vineyard::ObjectID string_id = 0x00000f00815840fc;
    auto r_int = std::dynamic_pointer_cast<vineyard::NumericArray<int64_t>>(client.GetObject(int_id));
    auto r_bool = std::dynamic_pointer_cast<vineyard::BooleanArray>(client.GetObject(bool_id));
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

    VLOG(0)<<"get bool array";
    
    auto bool_array = r_bool->GetArray();
    VLOG(0)<<"got bool array";
    for (int64_t i = 0; i < bool_array->length(); ++i) {
     std::cout<<bool_array->Value(i)<<std::endl;
    }
  // VLOG(0) << "this is level 0";
  // VLOG(2) << "this is level 2";
  // VLOG(4) << "this is level 4";
  return 0;
}

// void play_ipc(){
//     std::string ipc_socket = "/var/run/vineyard.sock";
//   vineyard::logging::InitGoogleLogging("vineyard");
//   vineyard::Client client;
//   VINEYARD_CHECK_OK(client.Connect(ipc_socket));
//   LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
  
//     vineyard::ObjectID bool_id = 0x0000044b814c2f84;
//     vineyard::ObjectID int_id = 0x000009922241b85e;
//     // vineyard::ObjectID string_id = 0x00000f00815840fc;
//     auto r_int = std::dynamic_pointer_cast<vineyard::NumericArray<arrow::Int64Array>>(client.GetObject(int_id));
//     auto r_bool = std::dynamic_pointer_cast<vineyard::BooleanArray>(client.GetObject(bool_id));
//     // vineyard::ObjectMeta  bool_meta;
//     // VINEYARD_CHECK_OK(client.GetMetaData(bool_id,bool_meta));
//     // VLOG(0)<< bool_meta.GetTypeName();
//     // auto r_string =  std::dynamic_pointer_cast<vineyard::NumericArray<std::string>>(client.GetObject(string_id));
//     // VLOG(0) << "get int array";
//     // auto int_array = r_int->GetArray();
//     // VLOG(0)<<"got int array";

//     // VLOG(0)<<"get string array";
//     // auto string_array = r_string->GetArray();
//     // VLOG(0)<<"got string array";

//     VLOG(0)<<"get bool array";
    
//     auto bool_array = r_bool->GetArray();
//     VLOG(0)<<"got bool array";
//     for (int64_t i = 0; i < bool_array->length(); ++i) {
//      std::cout<<bool_array->Value(i)<<std::endl;
//     }
//   // VLOG(0) << "this is level 0";
//   // VLOG(2) << "this is level 2";
//   // VLOG(4) << "this is level 4";
// }