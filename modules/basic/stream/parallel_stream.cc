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

#include "basic/stream/parallel_stream.h"

#include <memory>
#include <string>
#include <vector>

#include "client/client.h"

namespace vineyard {

// void ParallelStream::Construct(const ObjectMeta& meta) {
//   std::string __type_name = type_name<ParallelStream>();
//   VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
//                   "Expect typename '" + __type_name + "', but got '" +
//                       meta.GetTypeName() + "'");
//   this->meta_ = meta;
//   this->id_ = meta.GetId();

//   meta.GetKeyValue("size_", this->size_);
//   for (size_t idx = 0; idx < this->size_; ++idx) {
//     streams_.emplace_back(meta.GetMember("stream_" + std::to_string(idx)));
//   }
// }

// void ParallelStreamBuilder::AddStream(const ObjectID stream_id) {
//   streams_.emplace_back(stream_id);
// }

// Status ParallelStreamBuilder::Build(Client& client) { return Status::OK(); }

// std::shared_ptr<Object> ParallelStreamBuilder::_Seal(Client& client) {
//   // ensure the builder hasn't been sealed yet.
//   ENSURE_NOT_SEALED(this);

//   VINEYARD_CHECK_OK(this->Build(client));
//   auto __value = std::make_shared<ParallelStream>();

//   __value->meta_.SetTypeName(type_name<ParallelStream>());
//   __value->meta_.SetGlobal(true);

//   __value->size_ = streams_.size();
//   __value->meta_.AddKeyValue("size_", __value->size_);

//   for (size_t idx = 0; idx < streams_.size(); ++idx) {
//     __value->meta_.AddMember("stream_" + std::to_string(idx), streams_[idx]);
//   }
//   __value->meta_.SetNBytes(0);

//   VINEYARD_CHECK_OK(client.CreateMetaData(__value->meta_, __value->id_));
//   VINEYARD_CHECK_OK(client.GetMetaData(__value->id_, __value->meta_));

//   // mark the builder as sealed
//   VINEYARD_CHECK_OK(client.Persist(__value->id()));
//   this->set_sealed(true);

//   return std::static_pointer_cast<Object>(__value);
// }

}  // namespace vineyard
