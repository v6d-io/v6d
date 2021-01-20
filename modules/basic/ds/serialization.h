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

#ifndef MODULES_BASIC_DS_SERIALIZATION_H_
#define MODULES_BASIC_DS_SERIALIZATION_H_

#include "client/client.h"

namespace vineyard {

Status Serialize(Client& client, ObjectID in_id, ObjectID* stream_id);

Status deserialize(
    Client& client, const json& meta, ObjectMeta& target,
    const std::unordered_map<ObjectID, std::shared_ptr<Blob>>& blobs);

Status Deserialize(Client& client, ObjectID stream_id, ObjectID* out_id);

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_SERIALIZATION_H_
