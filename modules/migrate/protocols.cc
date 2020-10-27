/** Copyright 2020 Alibaba Group Holding Limited.

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

#include "migrate/protocols.h"

#include <iostream>
#include <sstream>

#include "boost/algorithm/string.hpp"

#include "common/util/boost.h"
#include "common/util/uuid.h"

namespace vineyard {

MigrateActionType ParseMigrateAction(const std::string& str_type) {
  if (str_type == "exit_request") {
    return MigrateActionType::ExitRequest;
  } else if (str_type == "send_object_request") {
    return MigrateActionType::SendObjectRequest;
  } else if (str_type == "send_blob_buffer_request") {
    return MigrateActionType::SendBlobBufferRequest;
  } else {
    return MigrateActionType::NullAction;
  }
}

static inline void encode_msg(const ptree& root, std::string& msg) {
  std::stringstream ss;
  bpt::write_json(ss, root, false);
  msg = ss.str();
}

void WriteErrorReply(Status const& status, std::string& msg) {
  encode_msg(status.ToJSON(), msg);
}

void WriteExitRequest(std::string& msg) {
  ptree root;
  root.put("type", "exit_request");

  encode_msg(root, msg);
}

void WriteSendObjectRequest(const ObjectID object_id, const ptree& object_meta,
                            std::string& msg) {
  ptree root;
  root.put("type", "send_object_request");
  root.put("id", object_id);
  root.add_child("meta", object_meta);
  encode_msg(root, msg);
}

Status ReadSendObjectRequest(const ptree& root, ObjectID& object_id,
                             ptree& object_meta) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "send_object_request");
  object_id = root.get<ObjectID>("id");
  object_meta = root.get_child("meta");
  return Status::OK();
}

void WriteSendBlobBufferRequest(const ObjectID blob_id, const size_t blob_size,
                                std::string& msg) {
  ptree root;
  root.put("type", "send_blob_buffer_request");
  root.put("blob_id", blob_id);
  root.put("blob_size", blob_size);
  encode_msg(root, msg);
}

Status ReadSendBlobBufferRequest(const ptree& root, ObjectID& blob_id,
                                 size_t& blob_size) {
  RETURN_ON_ASSERT(root.get<std::string>("type") == "send_blob_buffer_request");
  blob_id = root.get<ObjectID>("blob_id");
  blob_size = root.get<size_t>("blob_size");
  return Status::OK();
}

}  // namespace vineyard
