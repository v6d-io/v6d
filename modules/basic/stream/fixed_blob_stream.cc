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

#include <sys/mman.h>

#include <memory>
#include <string>
#include <vector>

#include "arrow/builder.h"
#include "arrow/status.h"

#include "basic/stream/fixed_blob_stream.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/stream.h"
#include "common/util/uuid.h"

namespace vineyard {

Status FixedBlobStream::Open(Client* client, StreamOpenMode mode, bool wait,
                             uint64_t timeout) {
  client_ = client;
  Status status = Status::OK();
  if (is_remote_) {
    status = client_->VineyardOpenRemoteFixedStream(
        stream_name_, this->id_, recv_mem_fd_, buffer_nums_, buffer_size_,
        rpc_endpoint_, mode, wait, timeout);
  } else {
    status = client_->OpenFixedStream(this->id_, mode, recv_mem_fd_);
  }
  if (status.ok()) {
    recv_flag_mem_ =
        mmap(0, STREAM_PAGE_SIZE, PROT_READ, MAP_SHARED, recv_mem_fd_, 0);
    if (recv_flag_mem_ == MAP_FAILED) {
      status = Status::IOError("Failed to mmap recv_flag_mem.");
      Close();
    }
  }
  return status;
}

Status FixedBlobStream::ActivateStreamWithBuffer(std::vector<void*>& buffers) {
  if (!is_remote_) {
    return Status::Invalid("Not a remote stream.");
  }
  return client_->VineyardActivateRemoteFixedStream(this->id_, buffers);
}

Status FixedBlobStream::ActivateStreamWithBlob(
    std::vector<ObjectID>& blob_list) {
  if (!is_remote_) {
    return Status::Invalid("Not a remote stream.");
  }
  return client_->VineyardActivateRemoteFixedStream(this->id_, blob_list);
}

Status FixedBlobStream::ActivateStreamWithOffset(
    std::vector<uint64_t>& offset_list) {
  if (!is_remote_) {
    return Status::Invalid("Not a remote stream.");
  }
  return client_->VineyardActivateRemoteFixedStreamWithOffset(this->id_,
                                                              offset_list);
}

Status FixedBlobStream::Push(uint64_t offset) {
  if (is_remote_) {
    return Status::Invalid("Cannot push to a remote stream.");
  }
  return client_->PushNextStreamChunkByOffset(this->id_, offset);
}

Status FixedBlobStream::CheckBlockReceived(int index, bool& finished) {
  unsigned char error_code = reinterpret_cast<unsigned char*>(
      recv_flag_mem_)[STREAM_PAGE_SIZE - sizeof(unsigned char)];
  std::string error_msg(reinterpret_cast<char*>(recv_flag_mem_) +
                            STREAM_PAGE_SIZE - STREAM_ERROR_LENGTH -
                            sizeof(unsigned char),
                        STREAM_ERROR_LENGTH);
  if (error_code != 0) {
    std::cerr << "Error code: " << static_cast<int>(error_code)
              << ", error message: " << error_msg << std::endl;
    Status status =
        Status(StatusCode(error_code), "Check block received failed.");
    return status;
  }

  if (index == -1) {
    for (int i = 0; i < buffer_nums_; ++i) {
      finished = true;
      if (reinterpret_cast<char*>(recv_flag_mem_)[i] == 0) {
        finished = false;
        break;
      }
    }
    return Status::OK();
  } else if (index >= 0 && index < buffer_nums_) {
    finished = (reinterpret_cast<char*>(recv_flag_mem_)[index] == 1);
    return Status::OK();
  } else {
    return Status::Invalid("Index out of range.");
  }
}

Status FixedBlobStream::Abort(bool& success) {
  if (is_remote_) {
    return client_->VineyardAbortRemoteStream(this->id_, success);
  } else {
    return client_->AbortStream(this->id_, success);
  }
}

Status FixedBlobStream::Close() {
  Status status = Status::OK();
  if (is_remote_) {
    status = client_->VineyardCloseRemoteFixedStream(this->id_);
  } else {
    status = client_->CloseStream(this->id_);
  }
  if (status.ok() && recv_flag_mem_ != nullptr) {
    munmap(recv_flag_mem_, STREAM_PAGE_SIZE);
    recv_flag_mem_ = nullptr;
  }
  client_ = nullptr;
  return status;
}

Status FixedBlobStream::Delete(Client* client, FixedBlobStream stream) {
  RETURN_ON_ERROR(client->DelData(stream.id_));
  return client->DeleteStream(stream.id_);
}

Status FixedBlobStream::PrintRecvInfo() {
  std::cout << "--------------------------" << std::endl;
  std::cout << " buffer_nums_: " << buffer_nums_ << std::endl;
  std::cout << "recv_mem_: " << recv_flag_mem_ << std::endl;
  for (int i = 0; i < buffer_nums_; ++i) {
    std::cout << "Recv flag " << i << ": "
              << static_cast<int>((reinterpret_cast<char*>(recv_flag_mem_))[i])
              << " " << std::endl;
  }
  std::cout << std::endl;
  std::cout << "--------------------------" << std::endl;
  return Status::OK();
}

}  // namespace vineyard
