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

#include "basic/stream/byte_stream.h"
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

static std::string base64_encode(const std::string& in) {
  std::string out;

  int val = 0, valb = -6;
  for (auto c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
              [(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6)
    out.push_back(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            [((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4)
    out.push_back('=');
  return out;
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
      if (kv.key() != "signature") {
        target.MutMetaData()[kv.key()] = kv.value();
      }
    }
  }
  ObjectID target_id = InvalidObjectID();
  RETURN_ON_ERROR(client.CreateMetaData(target, target_id));
  target.SetId(target_id);

  return Status::OK();
}

/// Restore meta and blobs from stream
/// If is a global object, then only restore local object
/// Return local object's id,
/// The caller is responsible for assemble objects into a global object.
Status Deserialize(Client& client, ObjectID stream_id, std::string& out_id) {
  auto byte_stream = client.GetObject<ByteStream>(stream_id);
  std::unique_ptr<ByteStreamReader> reader;
  RETURN_ON_ERROR(byte_stream->OpenReader(client, reader));

  // Consume meta
  auto params = byte_stream->GetParams();

  auto ordered_blobs =
      json::parse(params["blobs"]).get<std::vector<ObjectID>>();
  auto blobs_size =
      json::parse(params["blobs_size"]).get<std::vector<size_t>>();
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
  if (meta.value("global", false)) {
    auto sub_metas = json::parse(params["sub_metas"]).get<std::vector<json>>();
    std::unordered_map<std::string, std::string> target_id_map;
    // Reconstruct member's meta
    for (auto& sub_meta : sub_metas) {
      ObjectMeta sub_target;
      RETURN_ON_ERROR(deserialize_helper(client, sub_meta, sub_target, blobs));
      RETURN_ON_ERROR(client.Persist(sub_target.GetId()));
      target_id_map[sub_meta["id"].get<std::string>()] =
          sub_target.MetaData()["id"].get<std::string>();
      {
        ObjectMeta restored;
        RETURN_ON_ERROR(
            client.GetMetaData(sub_target.GetId(), restored, false));
        LOG(INFO) << "Target object type is " << sub_target.GetTypeName();
      }
    }
    std::stringstream ss;
    for (auto& pair : target_id_map) {
      ss << pair.first << ":" << pair.second << ";";
    }
    out_id = ss.str();
  } else {
    ObjectMeta target;
    RETURN_ON_ERROR(deserialize_helper(client, meta, target, blobs));
    RETURN_ON_ERROR(client.Persist(target.GetId()));
    {
      ObjectMeta restored;
      RETURN_ON_ERROR(client.GetMetaData(target.GetId(), restored, false));
      LOG(INFO) << "Target object type is " << target.GetTypeName();
    }
    out_id = target.MetaData()["id"].get<std::string>();
  }

  LOG(INFO) << "Deserialized from stream " << stream_id << " to object "
            << out_id;

  return Status::OK();
}

int main(int argc, const char** argv) {
  if (argc < 5) {
    printf(
        "usage ./deserializer <ipc_socket> <stream_id> <proc_num> "
        "<proc_index>\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  ObjectID stream_id = VYObjectIDFromString(argv[2]);
  int proc_num = std::stoi(argv[3]);
  int proc_index = std::stoi(argv[4]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto s =
      std::dynamic_pointer_cast<ParallelStream>(client.GetObject(stream_id));
  LOG(INFO) << "Got parallel stream " << s->id();

  auto ls = s->GetStream<ByteStream>(proc_index);
  LOG(INFO) << "Got byte stream " << ls->id() << " at " << proc_index << " (of "
            << proc_num << ")";
  if (proc_index == 0) {
    auto params = ls->GetParams();
    ReportStatus("return", base64_encode(params["meta"]));
  }
  std::string out;
  auto status = Deserialize(client, ls->id(), out);
  if (status.ok()) {
    ReportStatus("return", out);
    ReportStatus("exit", "");
  } else {
    ReportStatus("error", status.ToString());
  }
  return 0;
}
