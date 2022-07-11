/** Copyright 2020-2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <memory>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "basic/ds/sequence.h"
#include "common/util/logging.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/status.h"

using namespace vineyard; // NOLINT
using namespace std;  // NOLINT

void BasicTest(Client& client) {
  std::vector<double> double_array(250);
  // every double is 8bytes, insert 250 double
  for(int i = 0; i < 250; i++){
    double_array[i] = i;
  }
  ArrayBuilder<double> builder(client, double_array);
  auto sealed_double_array = std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
  ObjectID id1 = sealed_double_array->id();
  auto blob_id = ObjectIDFromString(sealed_double_array->meta().MetaData()["buffer_"]["id"]
                       .get_ref<std::string const&>());
  CHECK(blob_id != InvalidObjectID());
  CHECK(client.IsInUse(blob_id));
  VINEYARD_CHECK_OK(client.Release({id1, blob_id}));
  CHECK(!client.IsInUse(blob_id));
}

void ReloadTest(Client& client) {

}


int main(int argc, char** argv){
  if(argc < 2){
    printf("usage ./spill_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1, client2;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  VINEYARD_CHECK_OK(client2.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  BasicTest(client1);
  ReloadTest(client2);

  LOG(INFO) << "Passed spill tests ...";

  client1.Disconnect();
  client2.Disconnect();
}