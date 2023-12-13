/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#ifndef PYTHON_PYBIND11_DOCS_H_
#define PYTHON_PYBIND11_DOCS_H_

namespace vineyard {

namespace doc {

extern const char* ObjectID;
extern const char* ObjectName;

extern const char* ObjectMeta;
extern const char* ObjectMeta__init__;
extern const char* ObjectMeta_id;
extern const char* ObjectMeta_signature;
extern const char* ObjectMeta_typename;
extern const char* ObjectMeta_nbyte;
extern const char* ObjectMeta_instance_id;
extern const char* ObjectMeta_islocal;
extern const char* ObjectMeta_isglobal;
extern const char* ObjectMeta_set_global;
extern const char* ObjectMeta_memory_usage;
extern const char* ObjectMeta__contains__;
extern const char* ObjectMeta__getitem__;
extern const char* ObjectMeta_get;
extern const char* ObjectMeta_get_member;
extern const char* ObjectMeta__setitem__;
extern const char* ObjectMeta_add_member;

extern const char* Object;
extern const char* Object_id;
extern const char* Object_signature;
extern const char* Object_meta;
extern const char* Object_nbytes;
extern const char* Object_typename;
extern const char* Object_member;
extern const char* Object_islocal;
extern const char* Object_ispersist;
extern const char* Object_isglobal;

extern const char* Blob;
extern const char* Blob_size;
extern const char* Blob_is_empty;
extern const char* Blob_empty;
extern const char* Blob__len__;
extern const char* Blob_address;

extern const char* BlobBuilder;
extern const char* BlobBuilder_id;
extern const char* BlobBuilder__len__;
extern const char* BlobBuilder_abort;
extern const char* BlobBuilder_shrink;
extern const char* BlobBuilder_copy;
extern const char* BlobBuilder_address;

extern const char* RemoteBlob;
extern const char* RemoteBlob_id;
extern const char* RemoteBlob_instance_id;
extern const char* RemoteBlob_is_empty;
extern const char* RemoteBlob__len__;
extern const char* RemoteBlob_address;

extern const char* RemoteBlobBuilder;
extern const char* RemoteBlobBuilder_size;
extern const char* RemoteBlobBuilder_abort;
extern const char* RemoteBlobBuilder_copy;
extern const char* RemoteBlobBuilder_address;

extern const char* InstanceStatus;
extern const char* InstanceStatus_instance_id;
extern const char* InstanceStatus_deployment;
extern const char* InstanceStatus_memory_usage;
extern const char* InstanceStatus_memory_limit;
extern const char* InstanceStatus_deferred_requests;
extern const char* InstanceStatus_ipc_connections;
extern const char* InstanceStatus_rpc_connections;

extern const char* ClientBase;
extern const char* ClientBase_create_metadata;
extern const char* ClientBase_delete;
extern const char* ClientBase_persist;
extern const char* ClientBase_exists;
extern const char* ClientBase_shallow_copy;
extern const char* ClientBase_put_name;
extern const char* ClientBase_get_name;
extern const char* ClientBase_list_names;
extern const char* ClientBase_drop_name;
extern const char* ClientBase_sync_meta;
extern const char* ClientBase_clear;
extern const char* ClientBase_memory_trim;
extern const char* ClientBase_reset;
extern const char* ClientBase_connected;
extern const char* ClientBase_instance_id;
extern const char* ClientBase_meta;
extern const char* ClientBase_status;
extern const char* ClientBase_ipc_socket;
extern const char* ClientBase_rpc_endpoint;
extern const char* ClientBase_version;
extern const char* ClientBase_is_ipc;
extern const char* ClientBase_is_rpc;

extern const char* IPCClient;
extern const char* IPCClient_create_blob;
extern const char* IPCClient_create_empty_blob;
extern const char* IPCClient_get_blob;
extern const char* IPCClient_get_blobs;
extern const char* IPCClient_get_object;
extern const char* IPCClient_get_objects;
extern const char* IPCClient_get_meta;
extern const char* IPCClient_get_metas;
extern const char* IPCClient_list_objects;
extern const char* IPCClient_list_metadatas;
extern const char* IPCClient_allocated_size;
extern const char* IPCClient_is_shared_memory;
extern const char* IPCClient_find_shared_memory;
extern const char* IPCClient_close;

extern const char* RPCClient;
extern const char* RPCClient_get_object;
extern const char* RPCClient_get_objects;
extern const char* RPCClient_create_remote_blob;
extern const char* RPCClient_get_remote_blob;
extern const char* RPCClient_get_remote_blobs;
extern const char* RPCClient_get_meta;
extern const char* RPCClient_get_metas;
extern const char* RPCClient_list_objects;
extern const char* RPCClient_list_metadatas;
extern const char* RPCClient_close;
extern const char* RPCClient_is_fetchable;
extern const char* RPCClient_remote_instance_id;

extern const char* connect;

};  // namespace doc

}  // namespace vineyard

#endif  // PYTHON_PYBIND11_DOCS_H_
