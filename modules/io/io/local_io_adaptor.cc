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

#include "io/io/local_io_adaptor.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "boost/algorithm/string.hpp"
#include "glog/logging.h"

#include "basic/ds/arrow_utils.h"

namespace vineyard {
LocalIOAdaptor::LocalIOAdaptor(const std::string& location)
    : file_(nullptr),
      location_(location),
      using_std_getline_(false),
      header_row_(false),
      enable_partial_read_(false),
      total_parts_(0),
      index_(0) {
  // in csv format location:
  //    file_path#schema=t1,t2,t3&header_row=true/false

  // TODO: tidy with netlib for url parsing.
  size_t pos = location.find_first_of('#');
  if (pos != std::string::npos) {
    std::string config_field = location.substr(pos + 1);
    std::vector<std::string> config_list;
    // allows multiple # in configs
    ::boost::split(config_list, config_field, ::boost::is_any_of("&#"));
    for (auto& iter : config_list) {
      std::vector<std::string> kv_pair;
      ::boost::split(kv_pair, iter, ::boost::is_any_of("="));
      if (kv_pair[0] == "schema") {
        ::boost::split(columns_, kv_pair[1], ::boost::is_any_of(","));
        meta_.emplace("schema", kv_pair[1]);
      } else if (kv_pair[0] == "delimiter") {
        ::boost::algorithm::trim_if(kv_pair[1],
                                    boost::algorithm::is_any_of("\"\'"));
        if (kv_pair.size() > 1) {
          // handle escape character
          if (kv_pair[1][0] == '\\' && kv_pair[1][1] == 't') {
            delimiter_ = '\t';
          } else {
            delimiter_ = kv_pair[1][0];
          }
        }
        meta_.emplace("delimiter", std::string(1, delimiter_));
      } else if (kv_pair[0] == "header_row") {
        header_row_ = (kv_pair[1] == "true");
        meta_.emplace("header_row", std::to_string(header_row_));
      } else if (kv_pair.size() > 1) {
        meta_.emplace(kv_pair[0], kv_pair[1]);
      }
    }
    // Chech whether location contains protocol
    size_t begin_pos = 0;
    if (location_.substr(0, 7) == "file://") {
      begin_pos = 7;
    }
    location_ = location_.substr(begin_pos, pos - begin_pos);
  }
}

LocalIOAdaptor::~LocalIOAdaptor() {
  if (file_ != nullptr) {
    fclose(file_);
    file_ = nullptr;
  } else if (fs_.is_open()) {
    fs_.clear();
    fs_.close();
  }
}

Status LocalIOAdaptor::Open() { return this->Open("r"); }

Status LocalIOAdaptor::Open(const char* mode) {
  std::string tag = ".gz";
  size_t pos = location_.find(tag);
  if (pos != location_.size() - tag.size()) {
    if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
      int t = location_.find_last_of('/');
      if (t != -1) {
        std::string folder_path = location_.substr(0, t);
        if (access(folder_path.c_str(), 0) != 0) {
          RETURN_ON_ERROR(MakeDirectory(folder_path));
        }
      }
    }
    if (using_std_getline_) {
      if (strchr(mode, 'b') != NULL) {
        fs_.open(location_.c_str(),
                 std::ios::binary | std::ios::in | std::ios::out);
      } else if (strchr(mode, 'a') != NULL) {
        fs_.open(location_.c_str(),
                 std::ios::out | std::ios::in | std::ios::app);
      } else if (strchr(mode, 'w') != NULL || strchr(mode, '+') != NULL) {
        fs_.open(location_.c_str(),
                 std::ios::out | std::ios::in | std::ios::trunc);
      } else if (strchr(mode, 'r') != NULL) {
        fs_.open(location_.c_str(), std::ios::in);
      }
    } else {
      file_ = fopen(location_.c_str(), mode);
    }
  } else {
    return Status::NotImplemented();
  }

  if ((using_std_getline_ && !fs_) ||
      (!using_std_getline_ && file_ == nullptr)) {
    return Status::IOError("Failed to open the " + location_ +
                           " because: " + std::strerror(errno));
  }

  // check the partial read flag
  if (enable_partial_read_) {
    RETURN_ON_ERROR(setPartialReadImpl());
  } else {
    RETURN_ON_ERROR(ReadLine(header_line_));
    ::boost::algorithm::trim(header_line_);
    meta_.emplace("header_line", header_line_);
    ::boost::split(original_columns_, header_line_,
                   ::boost::is_any_of(std::string(1, delimiter_)));
  }
  return Status::OK();
}

Status LocalIOAdaptor::Configure(const std::string& key,
                                 const std::string& value) {
  if (key == "using_std_getline") {
    if (value == "false") {
      using_std_getline_ = false;
    } else if (value == "true") {
      using_std_getline_ = true;
    }
  }
  return Status::OK();
}

Status LocalIOAdaptor::SetPartialRead(const int index, const int total_parts) {
  // make sure that the bytes of each line of the file
  // is smaller than macro FINELINE
  if (index < 0 || total_parts <= 0 || index >= total_parts) {
    LOG(ERROR) << "error during set_partial_read with [" << index << ", "
               << total_parts << "]";
    return Status::IOError();
  }
  if (fs_.is_open() || file_ != nullptr) {
    LOG(WARNING) << "WARNING!! Set partial read after open have no effect,"
                    "You probably want to set partial before open!";
    return Status::IOError();
  }
  enable_partial_read_ = true;
  index_ = index;
  total_parts_ = total_parts;
  return Status::OK();
}

Status LocalIOAdaptor::GetPartialReadDetail(int64_t& offset, int64_t& nbytes) {
  if (!enable_partial_read_) {
    LOG(ERROR) << "Partial read is disabled, you probably want to"
                  " set partial read first.";
    return Status::IOError();
  }
  if ((using_std_getline_ && !fs_) ||
      (!using_std_getline_ && file_ == nullptr)) {
    LOG(ERROR) << "File not open, you probably want to open file first.";
    return Status::IOError();
  }
  offset = partial_read_offset_[index_];
  nbytes = partial_read_offset_[index_ + 1] - partial_read_offset_[index_];

  VLOG(2) << "partial read offset = " << offset << ", nbytes = " << nbytes;
  return Status::OK();
}

Status LocalIOAdaptor::setPartialReadImpl() {
  partial_read_offset_.resize(total_parts_ + 1,
                              std::numeric_limits<int>::max());
  partial_read_offset_[0] = 0;
  int start_pos = 0;
  if (header_row_) {
    RETURN_ON_ERROR(seek(0, kFileLocationBegin));
    RETURN_ON_ERROR(ReadLine(header_line_));
    ::boost::algorithm::trim(header_line_);
    ::boost::split(original_columns_, header_line_,
                   ::boost::is_any_of(std::string(1, delimiter_)));

    // skip header row
    int64_t dis = getDistanceToLineBreak(0);
    start_pos = dis + 1;
  }
  RETURN_ON_ERROR(seek(0, kFileLocationEnd));
  int64_t total_file_size = tell();
  if (start_pos > total_file_size) {
    start_pos = total_file_size;
  }
  int64_t part_size = (total_file_size - start_pos) / total_parts_;

  partial_read_offset_[0] = start_pos;
  partial_read_offset_[total_parts_] = total_file_size;

  // move breakpoint to the next of nearest character '\n'
  for (int i = 1; i < total_parts_; ++i) {
    partial_read_offset_[i] = i * part_size + start_pos;

    if (partial_read_offset_[i] < partial_read_offset_[i - 1]) {
      partial_read_offset_[i] = partial_read_offset_[i - 1];
    } else {
      // traversing backwards to find the nearest character '\n',
      int64_t dis = getDistanceToLineBreak(i);
      partial_read_offset_[i] += (dis + 1);
      if (partial_read_offset_[i] > total_file_size) {
        partial_read_offset_[i] = total_file_size;
      }
    }
  }

  int64_t file_stream_pos = partial_read_offset_[index_];
  RETURN_ON_ERROR(seek(file_stream_pos, kFileLocationBegin));
  return Status::OK();
}

Status LocalIOAdaptor::ReadTable(std::shared_ptr<arrow::Table>* table) {
  RETURN_ON_ERROR(ReadPartialTable(table, index_));
  return Status::OK();
}

Status LocalIOAdaptor::ReadPartialTable(std::shared_ptr<arrow::Table>* table,
                                        int index) {
  std::unique_ptr<arrow::fs::LocalFileSystem> arrow_lfs(
      new arrow::fs::LocalFileSystem());
  std::shared_ptr<arrow::io::RandomAccessFile> file_in;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(file_in,
                                   arrow_lfs->OpenInputFile(location_));

  int64_t offset = partial_read_offset_[index];
  int64_t nbytes =
      partial_read_offset_[index + 1] - partial_read_offset_[index];
  std::shared_ptr<arrow::io::InputStream> input =
      arrow::io::RandomAccessFile::GetStream(file_in, offset, nbytes);

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  if (!header_row_) {
    read_options.autogenerate_column_names = true;
  } else {
    read_options.column_names = original_columns_;
    if (!columns_.empty()) {
      convert_options.include_columns = columns_;
    }
  }
  parse_options.delimiter = delimiter_;

  std::shared_ptr<arrow::csv::TableReader> reader;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(pool, input, read_options,
                                            parse_options, convert_options));

  // RETURN_ON_ARROW_ERROR_AND_ASSIGN(*table, reader->Read());
  auto result = reader->Read();
  if (!result.status().ok()) {
    if (result.status().message() == "Empty CSV file") {
      *table = nullptr;
      return Status::OK();
    } else {
      return ::vineyard::Status::ArrowError(result.status());
    }
  }
  *table = result.ValueOrDie();

  RETURN_ON_ARROW_ERROR((*table)->Validate());

  VLOG(2) << "[file-" << location_ << "] contains: " << (*table)->num_rows()
          << " rows, " << (*table)->num_columns() << " columns";
  VLOG(2) << (*table)->schema()->ToString();
  return Status::OK();
}

int64_t LocalIOAdaptor::getDistanceToLineBreak(const int index) {
  VINEYARD_CHECK_OK(seek(partial_read_offset_[index], kFileLocationBegin));
  int64_t dis = 0;
  while (true) {
    char buffer[1];
    // std::memset(buff, 0, sizeof(buffer));
    auto status = Read(buffer, 1);
    if (!status.ok() || buffer[0] == '\n') {
      break;
    } else {
      dis++;
    }
  }
  return dis;
}

Status LocalIOAdaptor::ReadLine(std::string& line) {
  if (enable_partial_read_ && tell() >= partial_read_offset_[index_ + 1]) {
    return Status::EndOfFile();
  }
  if (using_std_getline_) {
    getline(fs_, line);
    if (line.empty()) {
      return Status::EndOfFile();
    } else {
      return Status::OK();
    }
  } else {
    if (file_ && fgets(buff, LINESIZE, file_)) {
      std::string str(buff);
      line.swap(str);
      return Status::OK();
    } else {
      return Status::EndOfFile();
    }
  }
}

Status LocalIOAdaptor::WriteLine(const std::string& line) {
  if (using_std_getline_) {
    if (!(fs_ << line)) {
      return Status::IOError();
    }
  } else {
    if (!file_ || fputs(line.c_str(), file_) <= 0) {
      return Status::IOError();
    }
  }
  return Status::OK();
}

Status LocalIOAdaptor::Seek(const int64_t offset) {
  return seek(offset, FileLocation::kFileLocationBegin);
}

int64_t LocalIOAdaptor::GetFullSize() {
  VINEYARD_CHECK_OK(seek(0, kFileLocationEnd));
  return tell();
}

int64_t LocalIOAdaptor::tell() {
  if (using_std_getline_) {
    return fs_.tellg();
  } else {
    return ftell(file_);
  }
}

Status LocalIOAdaptor::seek(const int64_t offset,
                            const FileLocation seek_from) {
  if (using_std_getline_) {
    fs_.clear();
    if (seek_from == kFileLocationBegin) {
      fs_.seekg(offset, fs_.beg);
    } else if (seek_from == kFileLocationCurrent) {
      fs_.seekg(offset, fs_.cur);
    } else if (seek_from == kFileLocationEnd) {
      fs_.seekg(offset, fs_.end);
    } else {
      return Status::Invalid();
    }
  } else {
    if (seek_from == kFileLocationBegin) {
      fseek(file_, offset, SEEK_SET);
    } else if (seek_from == kFileLocationCurrent) {
      fseek(file_, offset, SEEK_CUR);
    } else if (seek_from == kFileLocationEnd) {
      fseek(file_, offset, SEEK_END);
    } else {
      return Status::Invalid();
    }
  }
  return Status::OK();
}

Status LocalIOAdaptor::Read(void* buffer, size_t size) {
  if (using_std_getline_) {
    fs_.read(static_cast<char*>(buffer), size);
    if (!fs_) {
      return Status::EndOfFile();
    }
  } else {
    if (file_) {
      bool status = fread(buffer, 1, size, file_);
      if (!status) {
        return Status::EndOfFile();
      }
    } else {
      return Status::EndOfFile();
    }
  }
  return Status::OK();
}

Status LocalIOAdaptor::Write(void* buffer, size_t size) {
  if (using_std_getline_) {
    fs_.write(static_cast<char*>(buffer), size);
    if (!fs_) {
      return Status::IOError();
    }
    fs_.flush();
  } else {
    if (file_) {
      bool status = fwrite(buffer, 1, size, file_);
      if (!status) {
        return Status::IOError();
      }
      fflush(file_);
    } else {
      return Status::IOError();
    }
  }
  return Status::OK();
}

Status LocalIOAdaptor::Close() {
  if (using_std_getline_) {
    if (fs_.is_open()) {
      fs_.close();
    }
  } else {
    if (file_ != nullptr) {
      fclose(file_);
      file_ = nullptr;
    }
  }
  return Status::OK();
}

Status LocalIOAdaptor::ListDirectory(const std::string& path,
                                     std::vector<std::string>& files) {
  std::string s;
  DIR* dir;
  struct dirent* rent;
  dir = opendir(path.c_str());
  while ((rent = readdir(dir))) {
    s = rent->d_name;
    if (s[0] != '.') {
      files.push_back(s);
    }
  }
  closedir(dir);
  return Status::OK();
}

Status LocalIOAdaptor::MakeDirectory(const std::string& path) {
  std::string dir = path;
  int len = dir.size();
  if (dir[len - 1] != '/') {
    dir[len] = '/';
    len++;
  }
  std::string temp;
  for (int i = 1; i < len; i++) {
    if (dir[i] == '/') {
      temp = dir.substr(0, i);
      if (access(temp.c_str(), 0) != 0) {
        if (mkdir(temp.c_str(), 0777) != 0) {
          return Status::IOError();
        }
      }
    }
  }
  return Status::OK();
}

bool LocalIOAdaptor::IsExist(const std::string& path) {
  return access(location_.c_str(), 0) == 0;
}

void LocalIOAdaptor::Init() {}

void LocalIOAdaptor::Finalize() {}

}  // namespace vineyard
