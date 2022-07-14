#include "common/util/logging.h"
#include "client/client.h"
#include "basic/ds/arrow.h"
#include <iostream>
int main() {
  std::string ipc_socket = "/var/run/vineyard.sock";
  vineyard::logging::InitGoogleLogging("vineyard");
  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;
  
    vineyard::ObjectID bool_id = 0x00003d5da4eb2b08;
    vineyard::ObjectID int_id = 0x000009922241b85e;
    auto r_int = std::dynamic_pointer_cast<vineyard::NumericArray<int64_t>>(client.GetObject(int_id));
    auto r_bool = std::dynamic_pointer_cast<vineyard::NumericArray<bool>>(client.GetObject(bool_id));

    VLOG(0) << "get int array";
    auto int_array = r_int->GetArray();
    VLOG(0)<<"got int array";
    VLOG(0)<<"get bool array";
    auto bool_array = r_bool->GetArray();
    VLOG(0)<<"got bool array";
    // for (int64_t i = 0; i < internal_array->length(); ++i) {
    //  std::cout<<internal_array->Value(i)<<std::endl;
    // }
  // VLOG(0) << "this is level 0";
  // VLOG(2) << "this is level 2";
  // VLOG(4) << "this is level 4";
  return 0;
}