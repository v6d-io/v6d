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
  // in << meta.MetaData().dump();
  auto blobs = meta.GetBlobSet()->AllBlobIds();
  std::shared_ptr<Blob> blob = nullptr;
  size_t blob_size = 0;

  // Store blobs
  for (auto blob_id : blobs) {
    blob = client.GetObject<Blob>(blob_id);
    blob_size = blob->size();
    in << blob_id;
    in << blob_size;
    LOG(INFO) << "blob_id = " << VYObjectIDToString(blob_id) << " " << blob_size;
    if (blob_size != 0) {
#ifndef NDEBUG
      blob->Dump();
#endif
      in.AddBytes(blob->data(), blob_size);
    }
  }

  ByteStreamBuilder builder(client);
  builder.SetParam("meta", meta.MetaData().dump());
  *stream_id =
      std::dynamic_pointer_cast<ByteStream>(builder.Seal(client))->id();

  auto byte_stream = client.GetObject<ByteStream>(*stream_id);

  std::unique_ptr<ByteStreamWriter> writer;
  RETURN_ON_ERROR(byte_stream->OpenWriter(client, writer));
  // For simplicity, write whole buffer as a single chunk.
  // If this results in a memory issue in the future,
  // break the buffer up into pieces.
  LOG(INFO) << "archive size " << in.GetSize();
  RETURN_ON_ERROR(writer->WriteBytes(in.GetBuffer(), in.GetSize()));
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
  std::unordered_map<ObjectID, std::shared_ptr<Blob>> blobs;

  auto byte_stream = client.GetObject<ByteStream>(stream_id);
  std::unique_ptr<ByteStreamReader> reader;
  RETURN_ON_ERROR(byte_stream->OpenReader(client, reader));

  std::unique_ptr<arrow::Buffer> buffer = nullptr;
  RETURN_ON_ERROR(reader->GetNext(buffer));
  grape::OutArchive out;
  // The archive does not own the memory space.
  // Which means the `buffer` alive until the end.
  LOG(INFO) << "buffer size " << buffer->size();
  out.SetSlice(
      reinterpret_cast<char*>(const_cast<unsigned char*>(buffer->data())),
      buffer->size());
  {
    // Serialized buffer will have and only have one chunk.
    std::unique_ptr<arrow::Buffer> null_buffer;
    auto status = reader->GetNext(null_buffer);
    RETURN_ON_ASSERT(status.IsStreamDrained());
  }

  // Consume meta
  json meta = json::parse(byte_stream->GetParams()["meta"]);
  ObjectID blob_id;
  size_t blob_size;
  std::unique_ptr<BlobWriter> blob_writer;
  // Consume blob
  while (!out.Empty()) {
    out >> blob_id;
    out >> blob_size;
    LOG(INFO) << VYObjectIDToString(blob_id) << " " << blob_size;
    if (blob_size > 0) {
      RETURN_ON_ERROR(client.CreateBlob(blob_size, blob_writer));
      memcpy(blob_writer->data(), out.GetBytes(blob_size), blob_size);
#ifndef NDEBUG
      blob_writer->Dump();
#endif
      auto blob = std::dynamic_pointer_cast<Blob>(blob_writer->Seal(client));
#ifndef NDEBUG
      blob->Dump();
#endif
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