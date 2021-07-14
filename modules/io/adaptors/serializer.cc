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

#include "basic/stream/byte_stream.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

/// Put meta in streams' params,
/// And put all local blobs into ByteStream.
Status Serialize(Client& client, ObjectID in_id, ObjectID* stream_id) {
  ObjectMeta meta;
  RETURN_ON_ERROR(client.GetMetaData(in_id, meta, true));
  VLOG(10) << meta.MetaData().dump(4);
  std::vector<ObjectID> all_blobs;
  // Store meta
  ByteStreamBuilder builder(client);
  builder.SetParam("meta", meta.MetaData().dump());
  if (meta.IsGlobal()) {
    std::vector<json> sub_metas;
    for (const auto& kv : meta) {
      if (kv.value().is_object()) {
        ObjectMeta sub_meta = meta.GetMemberMeta(kv.key());
        if (sub_meta.GetInstanceId() == client.instance_id()) {
          sub_metas.push_back(sub_meta.MetaData());
          for (auto const& blob_id : sub_meta.GetBufferSet()->AllBufferIds()) {
            all_blobs.emplace_back(blob_id);
          }
        }
      }
    }
    builder.SetParam("sub_metas", json(sub_metas).dump());
  } else {
    for (auto const& blob_id : meta.GetBufferSet()->AllBufferIds()) {
      all_blobs.emplace_back(blob_id);
    }
  }

  std::vector<size_t> blobs_size;
  std::vector<std::shared_ptr<Blob>> blobs;
  for (auto blob_id : all_blobs) {
    blobs.push_back(client.GetObject<Blob>(blob_id));
    blobs_size.push_back(blobs.back()->size());
  }
  builder.SetParam("blobs", json(all_blobs).dump());
  builder.SetParam("blobs_size", json(blobs_size).dump());

  *stream_id =
      std::dynamic_pointer_cast<ByteStream>(builder.Seal(client))->id();
  VINEYARD_CHECK_OK(client.Persist(*stream_id));
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

int main(int argc, const char** argv) {
  if (argc < 3) {
    printf("usage ./serializer <ipc_socket> <object_id>\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  ObjectID object_id = VYObjectIDFromString(argv[2]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  ObjectID stream_id;
  auto st = Serialize(client, object_id, &stream_id);
  if (st.ok()) {
    ReportStatus("return", VYObjectIDToString(stream_id));
    ReportStatus("exit", "");
  } else {
    ReportStatus("error", st.ToString());
  }

  return 0;
}
