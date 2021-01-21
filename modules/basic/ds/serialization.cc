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

#include "basic/ds/serialization.h"

#include <unordered_map>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

#include "basic/stream/byte_stream.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

Status Serialize(Client& client, ObjectID in_id, ObjectID* stream_id) {
  grape::InArchive in;
  ObjectMeta meta;
  RETURN_ON_ERROR(client.GetMetaData(in_id, meta, true));
  // Store meta
  ByteStreamBuilder builder(client);
  builder.SetParam("meta", meta.MetaData().dump());

  auto blobs = meta.GetBlobSet()->AllBlobIds();
  std::vector<ObjectID> ordered_blobs(blobs.begin(), blobs.end());
  builder.SetParam("blobs", json(ordered_blobs).dump());

  *stream_id =
      std::dynamic_pointer_cast<ByteStream>(builder.Seal(client))->id();

  auto byte_stream = client.GetObject<ByteStream>(*stream_id);

  std::unique_ptr<ByteStreamWriter> writer;
  RETURN_ON_ERROR(byte_stream->OpenWriter(client, writer));
  // Store blobs
  std::shared_ptr<Blob> blob = nullptr;
  for (auto blob_id : ordered_blobs) {
    blob = client.GetObject<Blob>(blob_id);
    std::unique_ptr<arrow::MutableBuffer> buffer = nullptr;
    RETURN_ON_ERROR(writer->GetNext(blob->size(), buffer));
    memcpy(buffer->mutable_data(), blob->data(), blob->size());
  }
  RETURN_ON_ERROR(writer->Finish());
  LOG(INFO) << "Serialized object " << in_id << " to stream " << *stream_id;
  return Status::OK();
}

Status deserialize_helper(
    Client& client, const json& meta, ObjectMeta& target,
    const std::unordered_map<ObjectID, std::shared_ptr<Blob>>& blobs) {
  for (auto const& kv : meta.items()) {
    if (kv.value().is_object()) {
      auto member = kv.value();
      if (member["typename"].get_ref<std::string const&>() ==
          type_name<Blob>()) {
        target.AddMember(kv.key(),
                         blobs.at(VYObjectIDFromString(
                             member["id"].get_ref<std::string const&>())));
      } else {
        ObjectMeta sub_target;
        RETURN_ON_ERROR(deserialize_helper(client, member, sub_target, blobs));
        target.AddMember(kv.key(), sub_target);
      }
    } else {
      target.MutMetaData()[kv.key()] = kv.value();
    }
  }
  ObjectID target_id = InvalidObjectID();
  RETURN_ON_ERROR(client.CreateMetaData(target, target_id));
  target.SetId(target_id);

  return Status::OK();
}

Status Deserialize(Client& client, ObjectID stream_id, ObjectID* out_id) {
  auto byte_stream = client.GetObject<ByteStream>(stream_id);
  std::unique_ptr<ByteStreamReader> reader;
  RETURN_ON_ERROR(byte_stream->OpenReader(client, reader));

  // Consume meta
  json meta = json::parse(byte_stream->GetParams()["meta"]);

  std::vector<ObjectID> ordered_blobs = json::parse(byte_stream->GetParams()["blobs"]).get<std::vector<ObjectID>>();
  std::unique_ptr<BlobWriter> blob_writer;
  std::unordered_map<ObjectID, std::shared_ptr<Blob>> blobs;
  for (auto blob_id : ordered_blobs) {
    std::unique_ptr<arrow::Buffer> buffer = nullptr;
    RETURN_ON_ERROR(reader->GetNext(buffer));
    if (buffer->size() > 0) {
      RETURN_ON_ERROR(client.CreateBlob(buffer->size(), blob_writer));
      memcpy(blob_writer->data(), buffer->data(), buffer->size());
      auto blob = std::dynamic_pointer_cast<Blob>(blob_writer->Seal(client));
      blobs.emplace(blob_id, blob);
    } else {
      blobs.emplace(blob_id, Blob::MakeEmpty(client));
    }
  }

  ObjectMeta target;
  RETURN_ON_ERROR(deserialize_helper(client, meta, target, blobs));
  RETURN_ON_ERROR(client.Persist(target.GetId()));
  *out_id = target.GetId();
  {
    ObjectMeta restored;
    RETURN_ON_ERROR(client.GetMetaData(target.GetId(), restored, false));
    LOG(INFO) << "Target object type is " << target.GetTypeName();
  }
  LOG(INFO) << "Deserialized from stream " << stream_id << " to object "
            << *out_id;

  return Status::OK();
}

}  // namespace vineyard