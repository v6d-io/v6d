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

#include "migrate/object_migration.h"

#include <limits>
#include <map>
#include <string>
#include <utility>

#include "boost/asio.hpp"

#include "migrate/flags.h"
#include "migrate/protocols.h"

#define MAX_BUFFER_SIZE 1048576

namespace vineyard {

namespace asio = boost::asio;
using boost::asio::ip::tcp;

Status ObjectMigration::Migrate(
    std::unordered_map<InstanceID, InstanceID>& instance_map,
    std::unordered_map<ObjectID, InstanceID>& object_map, Client& client) {
  auto iter = instance_map.find(instance_id_);
  if (iter == instance_map.end()) {
    LOG(ERROR) << "Instance " << instance_id_ << " is not in the instance_map.";
    return Status::OK();
  }
  InstanceID target_instance = iter->second;
  std::string hostname;
  RETURN_ON_ERROR(getHostName(target_instance, client, hostname));
  boost::asio::io_service io_service;
  tcp::resolver resolver(io_service);
  tcp::resolver::query query(hostname, std::to_string(FLAGS_migration_port));
  tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
  tcp::endpoint endpoint = endpoint_iterator->endpoint();
  tcp::socket socket(io_service);
  socket.connect(endpoint);
  for (auto object_id : object_ids_) {
    LOG(INFO) << "Start send object " << object_id;
    RETURN_ON_ERROR(sendObjectMeta(object_id, client, socket));
  }
  for (auto blob_id : blob_list_) {
    std::shared_ptr<Blob> target_blob =
        std::dynamic_pointer_cast<Blob>(client.GetObject(blob_id));
    size_t remain_size = target_blob->size();
    std::string message_out;
    WriteSendBlobBufferRequest(blob_id, remain_size, message_out);
    size_t length = message_out.size();
    boost::asio::write(socket, asio::buffer(&length, sizeof(size_t)));
    boost::asio::write(socket, asio::buffer(message_out, message_out.size()));
    size_t offset = 0;
    while (remain_size) {
      size_t send_size =
          remain_size < MAX_BUFFER_SIZE ? remain_size : MAX_BUFFER_SIZE;
      boost::asio::write(socket,
                         asio::buffer(target_blob->data() + offset, send_size));
      remain_size -= send_size;
      offset += send_size;
    }
  }
  std::string message_exit;
  WriteExitRequest(message_exit);
  size_t length = message_exit.size();
  boost::asio::write(socket, asio::buffer(&length, sizeof(size_t)));
  boost::asio::write(socket, asio::buffer(message_exit, message_exit.size()));
  return Status::OK();
}

Status ObjectMigration::getHostName(InstanceID instance_id, Client& client,
                                    std::string& hostname) {
  std::map<uint64_t, json> cluster_info;
  RETURN_ON_ERROR(client.ClusterInfo(cluster_info));
  auto instance_info = cluster_info.find(instance_id);
  if (instance_info != cluster_info.end()) {
    auto& json = instance_info->second;
    hostname = json["hostname"].get_ref<std::string const&>();
  } else {
    LOG(ERROR) << "Dst instance id " << instance_id << " not exist";
    return Status::KeyError();
  }
  return Status::OK();
}

Status ObjectMigration::sendObjectMeta(ObjectID object_id, Client& client,
                                       tcp::socket& socket) {
  ObjectMeta object_meta;
  RETURN_ON_ERROR(client.GetMetaData(object_id, object_meta));
  json meta_tree = object_meta.MetaData();
  getBlobList(meta_tree);
  if (object_list_.find(object_id) == object_list_.end()) {
    std::string msg;
    WriteSendObjectRequest(object_id, meta_tree, msg);
    size_t length = msg.size();
    boost::asio::write(socket, asio::buffer(&length, sizeof(size_t)));
    boost::asio::write(socket, asio::buffer(msg, msg.size()));
    object_list_.emplace(object_id);
  }
  return Status::OK();
}

void ObjectMigration::getBlobList(json& meta_tree) {
  InstanceID instance_id = meta_tree["instance_id"].get<InstanceID>();
  if (instance_id != instance_id_)
    return;
  ObjectID id =
      ObjectIDFromString(meta_tree["id"].get_ref<std::string const&>());
  if (IsBlob(id) && blob_list_.find(id) == blob_list_.end()) {
    blob_list_.insert(id);
  }
  for (auto& item : meta_tree) {
    if (item.is_object() && !item.empty()) {
      getBlobList(item);
    }
  }
  return;
}

Status MigrationServer::Start(Client& client) {
  asio::io_service io_service;
  tcp::acceptor acceptor(io_service,
                         tcp::endpoint(tcp::v4(), FLAGS_migration_port));
  tcp::socket socket(io_service);
  acceptor.accept(socket);
  bool read_msg = true;
  while (read_msg) {
    size_t length;
    std::string message_in;
    boost::asio::read(socket, asio::buffer(&length, sizeof(size_t)));
    message_in.resize(length);
    boost::asio::read(socket, asio::buffer(&message_in[0], message_in.size()));
    json root = json::parse(message_in);

    std::string type = root["type"].get_ref<std::string const&>();
    MigrateActionType cmd = ParseMigrateAction(type);
    switch (cmd) {
    case MigrateActionType::SendObjectRequest: {
      LOG(INFO) << "cmd send object request";
      ObjectID object_id;
      json object_meta;
      RETURN_ON_ERROR(ReadSendObjectRequest(root, object_id, object_meta));
      // FIXME: this part will be abandoned when global object contains meta of
      // sub_object
      object_map_.emplace(object_id, object_meta);
    } break;
    case MigrateActionType::SendBlobBufferRequest: {
      ObjectID blob_id;
      size_t blob_size;
      RETURN_ON_ERROR(ReadSendBlobBufferRequest(root, blob_id, blob_size));
      std::unique_ptr<BlobWriter> buffer_writer;
      RETURN_ON_ERROR(client.CreateBlob(blob_size, buffer_writer));
      size_t remain_size = blob_size;
      size_t offset = 0;
      while (remain_size) {
        size_t recv_size =
            remain_size < MAX_BUFFER_SIZE ? remain_size : MAX_BUFFER_SIZE;
        boost::asio::read(
            socket, asio::buffer(buffer_writer->data() + offset, recv_size));
        remain_size -= recv_size;
        offset += recv_size;
      }
      auto buffer = buffer_writer->Seal(client);
      object_id_map_.emplace(blob_id, buffer->id());
    } break;
    case MigrateActionType::ExitRequest: {
      for (auto it = object_map_.begin(); it != object_map_.end(); it++) {
        ObjectID object_id;
        if (object_id_map_.find(it->first) == object_id_map_.end()) {
          if (it->second["instance_id"].get<InstanceID>() ==
              UnspecifiedInstanceID()) {
            object_id = createObject(it->second, client, true);
          } else {
            object_id = createObject(it->second, client, false);
          }
          object_id_map_.emplace(it->first, object_id);
        } else {
          object_id = object_id_map_.find(it->first)->second;
        }
        LOG(INFO) << "Build target local object " << it->first
                  << ", id: " << object_id;
      }
      read_msg = false;
      LOG(INFO) << "Migration server exit";
    } break;
    default: {
      LOG(ERROR) << "Got unexpected command: " << type;
    }
    }
  }
  return Status::OK();
}

ObjectID MigrationServer::createObject(json& meta_tree, Client& client,
                                       bool persist) {
  InstanceID instance_id = meta_tree["instance_id"].get<InstanceID>();
  for (auto& item : meta_tree.items()) {
    if (item.value().is_object() && !item.value().empty()) {
      ObjectID id =
          ObjectIDFromString(item.value()["id"].get_ref<std::string const&>());
      if (object_id_map_.find(id) == object_id_map_.end()) {
        InstanceID child_instance_id =
            item.value()["instance_id"].get<InstanceID>();
        std::string object_id =
            item.value()["id"].get_ref<std::string const&>();
        std::string query_name =
            object_id + "_" + std::to_string(child_instance_id);
        ObjectID new_object_id;
        if (child_instance_id == UnspecifiedInstanceID()) {
          new_object_id = createObject(item.value(), client, false);
        } else if (instance_map_.at(child_instance_id) ==
                   client.instance_id()) {
          new_object_id = createObject(item.value(), client,
                                       instance_id == UnspecifiedInstanceID());
        } else {
          VINEYARD_CHECK_OK(client.GetName(query_name, new_object_id, true));
          VINEYARD_CHECK_OK(client.DropName(query_name));
        }
        object_id_map_.emplace(id, new_object_id);
        id = new_object_id;
      } else {
        id = object_id_map_.find(id)->second;
      }
      ObjectMeta child_meta;
      VINEYARD_CHECK_OK(client.GetMetaData(id, child_meta));
      meta_tree[item.key()] = child_meta.MetaData();
    }
  }
  ObjectMeta new_meta;
  new_meta.SetMetaData(&client, meta_tree);
  if (instance_id != UnspecifiedInstanceID() ||
      client.instance_id() == instance_map_.begin()->second) {
    ObjectID obj_id;
    VINEYARD_CHECK_OK(client.CreateMetaData(new_meta, obj_id));
    VINEYARD_CHECK_OK(client.GetMetaData(obj_id, new_meta));
    if (persist) {
      if (instance_id != UnspecifiedInstanceID()) {
        std::string obj_name = ObjectIDToString(obj_id) + "_" +
                               std::to_string(client.instance_id());
        VINEYARD_CHECK_OK(client.PutName(obj_id, obj_name));
      }
      VINEYARD_CHECK_OK(client.Persist(obj_id));
    }
    return obj_id;
  } else {
    return InvalidObjectID();
  }
}

}  // namespace vineyard
