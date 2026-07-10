/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "common/util/logging.h"
#include "server/util/meta_tree.h"

namespace vineyard {
DEFINE_bool(prometheus, false, "Enable Prometheus metrics in the test");
}  // namespace vineyard

using namespace vineyard;  // NOLINT(build/namespaces)

namespace {

void CheckDeleteWithInvalidSignatureMetadata(const char* invalid_field,
                                             const json& invalid_value) {
  const std::string object_id = "o0000000000000001";
  json tree = {{"data",
                {{object_id,
                  {{"instance_id", static_cast<InstanceID>(7)},
                   {"signature", static_cast<Signature>(42)},
                   {"transient", true},
                   {"typename", "test::Object"}}}}}};
  tree["data"][object_id][invalid_field] = invalid_value;

  std::vector<meta_tree::op_t> ops;
  bool sync_remote = false;
  auto status = meta_tree::DelDataOps(tree, object_id, ops, sync_remote);

  CHECK(status.ok()) << status.ToString();
  CHECK_EQ(ops.size(), 1U);
  CHECK_EQ(ops[0].op, meta_tree::op_t::kDel);
  CHECK_EQ(ops[0].kv.key, "/data/" + object_id);
  CHECK(!sync_remote);
}

void CheckDeleteWithCompleteSignatureMetadata() {
  const std::string object_id = "o0000000000000002";
  json tree = {{"data",
                {{object_id,
                  {{"instance_id", static_cast<InstanceID>(7)},
                   {"signature", static_cast<Signature>(42)},
                   {"transient", false},
                   {"typename", "test::Object"}}}}}};

  std::vector<meta_tree::op_t> ops;
  bool sync_remote = false;
  auto status = meta_tree::DelDataOps(tree, object_id, ops, sync_remote);

  CHECK(status.ok()) << status.ToString();
  CHECK_EQ(ops.size(), 2U);
  CHECK_EQ(ops[0].kv.key, "/data/" + object_id);
  CHECK_EQ(ops[1].kv.key, "/signatures/i7/s000000000000002a");
  CHECK(sync_remote);
}

}  // namespace

int main(int argc, char** argv) {
  CheckDeleteWithInvalidSignatureMetadata("instance_id", nullptr);
  CheckDeleteWithInvalidSignatureMetadata("signature", nullptr);
  CheckDeleteWithInvalidSignatureMetadata("instance_id", "not-a-number");
  CheckDeleteWithInvalidSignatureMetadata("signature", "not-a-number");
  CheckDeleteWithCompleteSignatureMetadata();
  return 0;
}
