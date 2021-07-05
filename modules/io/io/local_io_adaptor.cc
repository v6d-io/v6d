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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/config.h"
#include "arrow/util/uri.h"

#include "boost/algorithm/string.hpp"
#include "glog/logging.h"

#include "basic/ds/arrow_utils.h"

namespace vineyard {
LocalIOAdaptor::LocalIOAdaptor(const std::string& location)
    : location_(location),
      header_row_(false),
      enable_partial_read_(false),
      total_parts_(0),
      index_(0) {
  // in csv format location:
  //    file_path#schema=t1,t2,t3&header_row=true/false

  // process the args
  //
  // TODO: tidy with netlib for url parsing.
  size_t arg_pos = location.find_first_of('#');
  if (arg_pos != std::string::npos) {
    // process arguments
    std::vector<std::string> config_list;
    // allows multiple # in configs
    std::string location_args = location.substr(arg_pos + 1);
    ::boost::split(config_list, location_args, ::boost::is_any_of("&#"));
    for (auto& iter : config_list) {
      std::vector<std::string> kv_pair;
      ::boost::split(kv_pair, iter, ::boost::is_any_of("="));
      if (kv_pair[0] == "schema") {
        ::boost::split(columns_, kv_pair[1], ::boost::is_any_of(","));
        meta_.emplace("schema", kv_pair[1]);
      } else if (kv_pair[0] == "column_types") {
        ::boost::split(column_types_, kv_pair[1], ::boost::is_any_of(","));
        meta_.emplace(kv_pair[0], kv_pair[1]);
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
        header_row_ = (boost::algorithm::to_lower_copy(kv_pair[1]) == "true");
        meta_.emplace("header_row", std::to_string(header_row_));
      } else if (kv_pair[0] == "include_all_columns") {
        include_all_columns_ =
            (boost::algorithm::to_lower_copy(kv_pair[1]) == "true");
        meta_.emplace("include_all_columns",
                      std::to_string(include_all_columns_));
      } else if (kv_pair.size() > 1) {
        meta_.emplace(kv_pair[0], kv_pair[1]);
      }
    }
  }

  // process locations
  location_ = location_.substr(0, arg_pos);
  size_t i = 0;
  for (i = 0; i < location_.size(); ++i) {
    if (location_[i] < 0 || location_[i] > 127) {
      break;
    }
  }
  auto encoded_location =
      location_.substr(0, i) + arrow::internal::UriEscape(location_.substr(i));
  fs_ = arrow::fs::FileSystemFromUriOrPath(encoded_location, &location_)
            .ValueOrDie();
#if defined(ARROW_VERSION) && ARROW_VERSION >= 3000000
  location_ = arrow::internal::UriUnescape(location_);
#else
  // Refered https://stackoverflow.com/questions/154536/encode-decode-urls-in-c
  auto urlDecode = [](std::string& src) -> std::string {
    std::string ret;
    char ch;
    int i = -1, ii = -1;
    for (i = 0; i < static_cast<int>(src.length()); i++) {
      if (src[i] == '%') {
        sscanf(src.substr(i + 1, 2).c_str(), "%x", &ii);
        ch = static_cast<char>(ii);
        ret += ch;
        i = i + 2;
      } else {
        ret += src[i];
      }
    }
    return ret;
  };
  location_ = urlDecode(location_);
#endif
}

LocalIOAdaptor::~LocalIOAdaptor() {
  VINEYARD_DISCARD(Close());
  fs_.reset();
}

std::unique_ptr<IIOAdaptor> LocalIOAdaptor::Make(const std::string& location,
                                                 Client* client) {
  // use `registered` to avoid it being optimized out.
  VLOG(999) << "Local IO adaptor has been registered: " << registered_;
  return std::unique_ptr<IIOAdaptor>(new LocalIOAdaptor(location));
}

Status LocalIOAdaptor::Open() { return this->Open("r"); }

Status LocalIOAdaptor::Open(const char* mode) {
  if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
    int t = location_.find_last_of('/');
    if (t != -1) {
      std::string folder_path = location_.substr(0, t);
      if (access(folder_path.c_str(), 0) != 0) {
        RETURN_ON_ERROR(MakeDirectory(folder_path));
      }
    }

    if (strchr(mode, 'w') != NULL) {
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(ofp_, fs_->OpenOutputStream(location_));
    } else {
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(ofp_, fs_->OpenAppendStream(location_));
    }
    return Status::OK();
  } else {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(ifp_, fs_->OpenInputFile(location_));

    // check the partial read flag
    if (enable_partial_read_) {
      RETURN_ON_ERROR(setPartialReadImpl());
    } else if (header_row_) {
      RETURN_ON_ERROR(ReadLine(header_line_));
      ::boost::algorithm::trim(header_line_);
      meta_.emplace("header_line", header_line_);
      ::boost::split(original_columns_, header_line_,
                     ::boost::is_any_of(std::string(1, delimiter_)));
    }
    return Status::OK();
  }
}

Status LocalIOAdaptor::Configure(const std::string& key,
                                 const std::string& value) {
  return Status::OK();
}

Status LocalIOAdaptor::SetPartialRead(const int index, const int total_parts) {
  // make sure that the bytes of each line of the file is smaller than macro
  // FINELINE
  if (index < 0 || total_parts <= 0 || index >= total_parts) {
    LOG(ERROR) << "error during set_partial_read with [" << index << ", "
               << total_parts << "]";
    return Status::IOError();
  }
  if (ifp_ != nullptr) {
    LOG(WARNING) << "WARNING!! Set partial read after open have no effect, "
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
    LOG(ERROR) << "Partial read is disabled, you probably want to "
                  "set partial read first.";
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
    meta_.emplace("header_line", header_line_);
    ::boost::split(original_columns_, header_line_,
                   ::boost::is_any_of(std::string(1, delimiter_)));

    // skip header row
    int64_t dis = getDistanceToLineBreak(0);
    start_pos = dis + 1;
  } else {
    // Name columns as f0 ... fn
    std::string one_line;
    RETURN_ON_ERROR(seek(0, kFileLocationBegin));
    RETURN_ON_ERROR(ReadLine(one_line));
    ::boost::algorithm::trim(one_line);
    meta_.emplace("header_line", one_line);
    std::vector<std::string> one_column;
    ::boost::split(one_column, one_line,
                   ::boost::is_any_of(std::string(1, delimiter_)));
    for (size_t i = 0; i < one_column.size(); ++i) {
      original_columns_.push_back("f" + std::to_string(i));
    }
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

/// \a origin_columns_ saves the column names of the CSV.
///
/// If \a header_row == \a true, \a origin_columns will be read from the first
/// CSV row. If \a header_row == \a false, \a origin_columns will be of the form
/// "f0", "f1", ...
///
/// Assume the order of \a column_types is same with \a include_columns.
/// For example:
/// \a include_columns: a,b,c,d
/// \a column_types   : int,double,float,string
/// Additionally, \a include_columns may have numbers, like "0,1,c,d"
/// The number means index in \a origin_columns.
/// We only use numbers for vid index.
/// So we should get the name from \a origin_columns, then associate it with
/// column type.

/// \a column_types also may have empty fields, means let arrow deduce type
/// for that column.
/// For example:
///     column_types: int,,,string.
/// Means we deduce the type of the second and third column.
Status LocalIOAdaptor::ReadPartialTable(std::shared_ptr<arrow::Table>* table,
                                        int index) {
  if (ifp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in read mode: " +
                           location_);
  }
  int64_t offset = partial_read_offset_[index];
  int64_t nbytes =
      partial_read_offset_[index + 1] - partial_read_offset_[index];
  std::shared_ptr<arrow::io::InputStream> input =
      arrow::io::RandomAccessFile::GetStream(ifp_, offset, nbytes);

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  read_options.column_names = original_columns_;

  auto is_number = [](const std::string& s) -> bool {
    return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                           return !std::isdigit(c);
                         }) == s.end();
  };

  // Get all indices represented column, and get their name
  std::vector<int> indices;
  for (size_t i = 0; i < columns_.size(); ++i) {
    if (is_number(columns_[i])) {
      int col_idx = std::stoi(columns_[i]);
      if (col_idx >= static_cast<int>(original_columns_.size())) {
        return Status(StatusCode::kArrowError,
                      "Index out of range: " + columns_[i]);
      }
      indices.push_back(col_idx);
      columns_[i] = original_columns_[col_idx];
    }
  }

  // If include_all_columns_ is set, push other names as well
  if (include_all_columns_) {
    for (const auto& col : original_columns_) {
      if (std::find(std::begin(columns_), std::end(columns_), col) ==
          columns_.end()) {
        columns_.push_back(col);
      }
    }
  }

  convert_options.include_columns = columns_;

  if (column_types_.size() > convert_options.include_columns.size()) {
    return Status(StatusCode::kArrowError,
                  "Format of column type schema is incorrect.");
  }
  std::unordered_map<std::string, std::shared_ptr<arrow::DataType>>
      column_types;

  for (size_t i = 0; i < column_types_.size(); ++i) {
    if (!column_types_[i].empty()) {
      column_types[convert_options.include_columns[i]] =
          type_name_to_arrow_type(column_types_[i]);
    }
  }
  convert_options.column_types = column_types;

  parse_options.delimiter = delimiter_;

  std::shared_ptr<arrow::csv::TableReader> reader;
#if defined(ARROW_VERSION) && ARROW_VERSION >= 4000000
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(arrow::io::IOContext(pool), input,
                                            read_options, parse_options,
                                            convert_options));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(pool, input, read_options,
                                            parse_options, convert_options));
#endif

  // RETURN_ON_ARROW_ERROR_AND_ASSIGN(*table, reader->Read());
  auto result = reader->Read();
  if (!result.status().ok()) {
    if (result.status().message() == "Empty CSV file") {
      *table = nullptr;
      return Status::OK();
    } else {
      return Status::ArrowError(result.status());
    }
  }
  *table = result.ValueOrDie();

  RETURN_ON_ARROW_ERROR((*table)->Validate());

  VLOG(2) << "[file-" << location_ << "] contains: " << (*table)->num_rows()
          << " rows, " << (*table)->num_columns() << " columns";
  VLOG(2) << (*table)->schema()->ToString();
  return Status::OK();
}

// TODO: sub-optimal, requires further optimization
int64_t LocalIOAdaptor::getDistanceToLineBreak(const int index) {
  VINEYARD_CHECK_OK(seek(partial_read_offset_[index], kFileLocationBegin));

  constexpr int64_t nbytes_for_block = 255;
  char buffer[256];

  int64_t dis = 0;
  while (true) {
    auto sz = ifp_->Read(nbytes_for_block, buffer);
    if (sz.ok() && sz.ValueUnsafe() > 0) {
      int64_t read_size = sz.ValueUnsafe();

      buffer[read_size] = '\0';
      char* endofline = strchr(buffer, '\n');
      if (endofline != nullptr) {
        dis += endofline - buffer;  // points to previous char before the `\n`.
        return dis;
      } else {
        dis += read_size;
      }
    } else {
      return dis;
    }
  }
}

// TODO: sub-optimal, requires further optimization
Status LocalIOAdaptor::ReadLine(std::string& line) {
  if (ifp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in read mode: " +
                           location_);
  }
  if (enable_partial_read_ && tell() >= partial_read_offset_[index_ + 1]) {
    return Status::EndOfFile();
  }

  constexpr int64_t nbytes_for_block = 256;
  size_t offset = 0, skip_endofline = 0;
  int64_t start_position = ifp_->Tell().ValueOrDie();

  // find delimiter
  while (true) {
    auto sz = ifp_->Read(nbytes_for_block, buff + offset);
    if (sz.ok() && sz.ValueUnsafe() > 0) {
      int64_t read_size = sz.ValueUnsafe();

      // find '\n'
      VINEYARD_ASSERT(offset + read_size < LINESIZE - 1,
                      "The line is too long that is not supported");
      buff[offset + read_size] = '\0';
      char* endofline = strchr(buff + offset, '\n');
      if (endofline != nullptr) {
        offset = endofline - buff;
        skip_endofline = 1;
        break;
      } else {
        offset += read_size;
      }
    } else {
      if (offset == 0) {
        return Status::EndOfFile();
      }
      break;
    }
  }

  VINEYARD_DISCARD(
      Status::ArrowError(ifp_->Seek(start_position + offset + skip_endofline)));
  std::string linebuffer = std::string(buff, offset);
  line.swap(linebuffer);
  return Status::OK();
}

Status LocalIOAdaptor::WriteLine(const std::string& line) {
  if (ofp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in write mode: " +
                           location_);
  }
  auto status = ofp_->Write(line.c_str(), line.size());
  if (status.ok()) {
    return Status::ArrowError(ofp_->Write("\n", 1));
  } else {
    return Status::ArrowError(status);
  }
}

Status LocalIOAdaptor::Seek(const int64_t offset) {
  return seek(offset, FileLocation::kFileLocationBegin);
}

int64_t LocalIOAdaptor::GetFullSize() {
  if (ifp_) {
    return ifp_->GetSize().ValueOr(-1);
  }
  return -1;
}

int64_t LocalIOAdaptor::tell() {
  if (ifp_) {
    return ifp_->Tell().ValueOr(-1);
  }
  if (ofp_) {
    return ofp_->Tell().ValueOr(-1);
  }
  return -1;
}

Status LocalIOAdaptor::seek(const int64_t offset,
                            const FileLocation seek_from) {
  if (!ifp_) {
    return Status::Invalid("Not a seekable random access file: " + location_);
  }
  switch (seek_from) {
  case kFileLocationBegin: {
    return Status::ArrowError(ifp_->Seek(offset));
  } break;
  case kFileLocationCurrent: {
    auto p = ifp_->Tell();
    if (p.ok()) {
      return Status::ArrowError(ifp_->Seek(p.ValueUnsafe() + offset));
    } else {
      return Status::IOError("Fail to tell current position: " + location_);
    }
  } break;
  case kFileLocationEnd: {
    auto sz = ifp_->GetSize();
    if (sz.ok()) {
      return Status::ArrowError(ifp_->Seek(sz.ValueUnsafe() - offset));
    } else {
      return Status::IOError("Fail to tell the total file size: " + location_);
    }
  } break;
  default: {
    return Status::Invalid("Not support seek mode: " +
                           std::to_string(static_cast<int>(seek_from)));
  }
  }
}

Status LocalIOAdaptor::Read(void* buffer, size_t size) {
  if (ifp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in read mode: " +
                           location_);
  }
  auto r = ifp_->Read(size, buffer);
  if (r.ok()) {
    if (r.ValueUnsafe() < static_cast<int64_t>(size)) {
      return Status::EndOfFile();
    } else {
      return Status::OK();
    }
  } else {
    return Status::ArrowError(r.status());
  }
}

Status LocalIOAdaptor::Write(void* buffer, size_t size) {
  if (ofp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in write mode: " +
                           location_);
  }
  return Status::ArrowError(ofp_->Write(buffer, size));
}

Status LocalIOAdaptor::Flush() {
  if (ofp_ == nullptr) {
    return Status::IOError("The file hasn't been opened in write mode: " +
                           location_);
  }
  return Status::ArrowError(ofp_->Flush());
}

Status LocalIOAdaptor::Close() {
  Status s1, s2;
  if (ifp_) {
    s1 = Status::ArrowError(ifp_->Close());
  }
  if (ofp_) {
    auto status = ofp_->Flush();
    if (status.ok()) {
      s2 = Status::ArrowError(ofp_->Close());
    } else {
      s2 = Status::ArrowError(status);
    }
  }
  return s1 & s2;
}

Status LocalIOAdaptor::ListDirectory(const std::string& path,
                                     std::vector<std::string>& files) {
  arrow::fs::FileSelector selector;
  selector.base_dir = path;
  std::vector<arrow::fs::FileInfo> infos;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(infos, fs_->GetFileInfo(selector));
  for (auto const& finfo : infos) {
    files.emplace_back(finfo.path());
  }
  return Status::OK();
}

Status LocalIOAdaptor::MakeDirectory(const std::string& path) {
  return Status::ArrowError(fs_->CreateDir(path, true));
}

bool LocalIOAdaptor::IsExist(const std::string& path) {
  auto mfinfo = fs_->GetFileInfo(path);
  return mfinfo.ok() &&
         mfinfo.ValueUnsafe().type() != arrow::fs::FileType::NotFound;
}

void LocalIOAdaptor::Init() {}

void LocalIOAdaptor::Finalize() {}

const bool LocalIOAdaptor::registered_ = IOFactory::Register(
    {"file", "hdfs", "s3"},
    static_cast<IOFactory::io_initializer_t>(&LocalIOAdaptor::Make));

}  // namespace vineyard
