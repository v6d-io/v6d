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

#include "basic/stream/byte_stream.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

Status Serialize(Client& client, ObjectID in_id, ObjectID* stream_id) {
  ObjectMeta meta;
  RETURN_ON_ERROR(client.GetMetaData(in_id, meta, true));
  VLOG(10) << meta.MetaData().dump(4);
  std::vector<ObjectID> ordered_blobs;
  // Store meta
  ByteStreamBuilder builder(client);
  builder.SetParam("meta", meta.MetaData().dump());
  if (meta.IsGlobal()) {
    std::vector<json> sub_metas;
    for (const auto& kv : meta) {
      if (kv.value().is_object()) {
        auto member = kv.value();
        ObjectID sub_id = VYObjectIDFromString(member["id"].get_ref<std::string const&>());
        ObjectMeta sub_meta;
        RETURN_ON_ERROR(client.GetMetaData(sub_id, sub_meta, true));
        sub_metas.push_back(sub_meta.MetaData());

        auto blob_ids = sub_meta.GetBlobSet()->AllBlobIds();
        ordered_blobs.insert(ordered_blobs.end(), blob_ids.begin(), blob_ids.end());
      }
    }
    builder.SetParam("sub_metas", json(sub_metas).dump());
  } else {
    auto blob_ids = meta.GetBlobSet()->AllBlobIds();
    ordered_blobs.insert(ordered_blobs.end(), blob_ids.begin(), blob_ids.end());
  }

  std::vector<size_t> blobs_size;
  std::vector<std::shared_ptr<Blob>> blobs;
  for (auto blob_id : ordered_blobs) {
    blobs.push_back(client.GetObject<Blob>(blob_id));
    blobs_size.push_back(blobs.back()->size());
  }
  builder.SetParam("blobs", json(ordered_blobs).dump());
  builder.SetParam("blobs_size", json(blobs_size).dump());

  *stream_id =
      std::dynamic_pointer_cast<ByteStream>(builder.Seal(client))->id();

  auto byte_stream = client.GetObject<ByteStream>(*stream_id);

  std::unique_ptr<ByteStreamWriter> writer;
  RETURN_ON_ERROR(byte_stream->OpenWriter(client, writer));
  // Store blobs
  for (const auto& blob : blobs) {
    if (blob->size() > 0) {
      std::unique_ptr<arrow::MutableBuffer> buffer = nullptr;
      RETURN_ON_ERROR(writer->GetNext(blob->size(), buffer));
      memcpy(buffer->mutable_data(), blob->data(), blob->size());
    }
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
  auto params = byte_stream->GetParams();

  auto ordered_blobs = json::parse(params["blobs"]).get<std::vector<ObjectID>>();
  auto blobs_size = json::parse(params["blobs_size"]).get<std::vector<size_t>>();
  std::unique_ptr<BlobWriter> blob_writer;
  std::unordered_map<ObjectID, std::shared_ptr<Blob>> blobs;
  for (size_t i = 0; i < ordered_blobs.size(); ++i) {
    if (blobs_size[i] > 0) {
      std::unique_ptr<arrow::Buffer> buffer = nullptr;
      RETURN_ON_ERROR(reader->GetNext(buffer));
      RETURN_ON_ERROR(client.CreateBlob(buffer->size(), blob_writer));
      memcpy(blob_writer->data(), buffer->data(), buffer->size());
      auto blob = std::dynamic_pointer_cast<Blob>(blob_writer->Seal(client));
      blobs.emplace(ordered_blobs[i], blob);
    } else {
      blobs.emplace(ordered_blobs[i], Blob::MakeEmpty(client));
    }
  }
  json meta = json::parse(params["meta"]);
  ObjectMeta target;
  if (meta.value("global", false)) {
    auto sub_metas = json::parse(params["sub_metas"]).get<std::vector<json>>();
    std::unordered_map<ObjectID, ObjectID> target_id_map;
    // Reconstruct all member's meta
    for (auto &sub_meta : sub_metas) {
      ObjectMeta sub_target;
      RETURN_ON_ERROR(deserialize_helper(client, sub_meta, sub_target, blobs));
      RETURN_ON_ERROR(client.Persist(sub_target.GetId()));
      auto old_id = VYObjectIDFromString(sub_meta["id"].get_ref<std::string const&>());
      target_id_map[old_id] = sub_target.GetId();
    }
    // Replace old member ID to newly constructed
    for (const auto& kv : meta.items()) {
      if (kv.value().is_object()) {
        auto member = kv.value();
        auto old_id = VYObjectIDFromString(member["id"].get_ref<std::string const&>());
        target.AddMember(kv.key(), target_id_map[old_id]);
      } else {
        target.MutMetaData()[kv.key()] = kv.value();
      }
    }
  } else {
    RETURN_ON_ERROR(deserialize_helper(client, meta, target, blobs));
  }

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