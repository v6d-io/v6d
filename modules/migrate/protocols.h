/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#ifndef MODULES_MIGRATE_PROTOCOLS_H_
#define MODULES_MIGRATE_PROTOCOLS_H_

#include <string>

#include "common/util/json.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

enum class MigrateActionType {
  NullAction = 0,
  ExitRequest = 1,
  ExitReply = 2,
  SendObjectRequest = 3,
  SendBlobBufferRequest = 4,
};

MigrateActionType ParseMigrateAction(const std::string& str_type);

void WriteErrorReply(Status const& status, std::string& msg);

void WriteExitRequest(std::string& msg);

void WriteSendObjectRequest(const ObjectID object_id, const json& object_meta,
                            std::string& msg);

Status ReadSendObjectRequest(const json& root, ObjectID& object_id,
                             json& object_meta);

void WriteSendBlobBufferRequest(const ObjectID blob_id, const size_t blob_size,
                                std::string& msg);

Status ReadSendBlobBufferRequest(const json& root, ObjectID& blob_id,
                                 size_t& blob_size);

}  // namespace vineyard

#endif  // MODULES_MIGRATE_PROTOCOLS_H_
