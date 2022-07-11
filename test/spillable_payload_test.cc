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

#include <glog/logging.h>
#include <memory>
#include <vector>
#include "common/util/status.h"
#include "server/memory/memory.h"
#include "server/memory/spillable_payload.h"

using namespace vineyard;     // NOLINT
using namespace std;          // NOLINT

constexpr size_t mem_limit = 2048 + 16;

std::shared_ptr<BulkStore> InitBulkStore() {
  std::shared_ptr<BulkStore> bulk_store_ptr = make_shared<BulkStore>();
  VINEYARD_CHECK_OK(bulk_store_ptr->PreAllocate(mem_limit));
  bulk_store_ptr->SetMemSpillSize(mem_limit/3);
  return bulk_store_ptr;
}

void BasicTest() {
  auto bulk_store_ptr = InitBulkStore();
  std::shared_ptr<Payload> pl_ptr;
  ObjectID pl_id;
  VINEYARD_CHECK_OK(bulk_store_ptr->Create(2000, pl_id, pl_ptr));
  cout << "Object ID: " << pl_id << endl;
  memcpy(pl_ptr->pointer, "Basictest", 10);
  VINEYARD_CHECK_OK(bulk_store_ptr->Seal(pl_id));
  // before release
  {
    std::shared_ptr<Payload> pl_ptr2;
    ObjectID pl_id2;
    CHECK_EQ(bulk_store_ptr->Create(1024, pl_id2, pl_ptr2).IsNotEnoughMemory(), true);
  }
  // after release
  VINEYARD_CHECK_OK(bulk_store_ptr->MarkAsCold(pl_id, pl_ptr));
  {
    bool is_in_use{false};
    VINEYARD_CHECK_OK(bulk_store_ptr->IsInUse(pl_id, is_in_use));
    CHECK_EQ(is_in_use, false);
  }
  {
    std::shared_ptr<Payload> pl_ptr2;
    ObjectID pl_id2 = 2;
    VINEYARD_CHECK_OK(bulk_store_ptr->Create(2000, pl_id2, pl_ptr2));
  }
  {
    bool is_spilled{false};
    VINEYARD_CHECK_OK(bulk_store_ptr->IsSpilled(pl_id, is_spilled));
    CHECK_EQ(is_spilled, true);
    CHECK_EQ(pl_ptr->pointer == nullptr, true);
  }
}

void ReloadTest(){
  auto bulk_store_ptr = InitBulkStore();
  std::shared_ptr<Payload> pl_ptr;
  ObjectID pl_id;
  VINEYARD_CHECK_OK(bulk_store_ptr->Create(2000, pl_id, pl_ptr));
  memcpy(pl_ptr->pointer, "ReloadTest", 10);
  VINEYARD_CHECK_OK(bulk_store_ptr->Seal(pl_id));
  VINEYARD_CHECK_OK(bulk_store_ptr->MarkAsCold(pl_id, pl_ptr));
  {
    bool is_in_use{false};
    VINEYARD_CHECK_OK(bulk_store_ptr->IsInUse(pl_id, is_in_use));
    CHECK_EQ(is_in_use, false);
  }
  std::shared_ptr<Payload> pl_ptr2;
  ObjectID pl_id2 = 2;
  VINEYARD_CHECK_OK(bulk_store_ptr->Create(2000, pl_id2, pl_ptr2));
  LOG(INFO) << "Object ID1 : " << pl_id << ", Object ID2 : " << pl_id2 << endl;
  {
    bool is_spilled{false};
    VINEYARD_CHECK_OK(bulk_store_ptr->IsSpilled(pl_id, is_spilled));
    CHECK_EQ(is_spilled, true);
  }
  // release payload2
  VINEYARD_CHECK_OK(bulk_store_ptr->Seal(pl_id2));
  // then try to reload payload1
  LOG(INFO) << "Now RemoveFromColdList test";
  {
    CHECK_EQ(bulk_store_ptr->RemoveFromColdList(pl_id).IsNotEnoughMemory(), true);
  }
  {
    VINEYARD_CHECK_OK(bulk_store_ptr->MarkAsCold(pl_id2, pl_ptr2));
    VINEYARD_CHECK_OK(bulk_store_ptr->RemoveFromColdList(pl_id));
    CHECK_EQ(pl_ptr->pointer == nullptr, false);
    CHECK_EQ(memcmp(pl_ptr->pointer, "ReloadTest", 10), 0);
  }
  LOG(INFO) << "Finished";
}

int main(int argc, char** argv) {
  LOG(INFO) << "----------BasicTest()------------" << endl;
  BasicTest();
  LOG(INFO) << "----------ReloadTest()------------" << endl;
  ReloadTest();
  LOG(INFO) << "Passed spillable payload tests...";
  return 0;
}