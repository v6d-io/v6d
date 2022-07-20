/* #include "arrow/array.h"
#include "arrow/buffer.h"

 std::shared_ptr<arrow::Buffer> vineyard::fuse::NumericArrayAdaptor::Deserialize(
                    std::shared_ptr<vineyard::NumericArray<int64_t>>& arr) {
                    VLOG(2)<< "arrow_ipc_view is called";
                    std::shared_ptr<arrow::io::BufferOutputStream> ssink;

                FUSE_ASSIGN_OR_RAISE( ssink, arrow::io::BufferOutputStream::Create(
                                    ));
                    VLOG(2)<<"buffer successfully created";

                auto kvmeta = std::shared_ptr<arrow::KeyValueMetadata>(
                    new arrow::KeyValueMetadata({}, {}));
                    
                auto meta = arr->meta();
                
                for (auto i : meta) {
                    std::string v = i.value().dump();
                    kvmeta->Append(i.key(), v);
                }
                auto schema = arrow::schema(
                    {arrow::field("a", ConvertToArrowType<int64_t>::TypeValue())}, kvmeta);
                std::shared_ptr<arrow::Table> my_table =
                    arrow::Table::Make(schema, {arr->GetArray()});
                    
                std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
                CHECK_ARROW_ERROR_AND_ASSIGN( writer,
                                            arrow::ipc::MakeStreamWriter(ssink, schema));
                
                VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
                VLOG(4)<< "table is written";
                writer->Close();
                    std::shared_ptr<arrow::Buffer> buffer_;
                VLOG(3)<< "writer is closed";
                CHECK_ARROW_ERROR_AND_ASSIGN( buffer_,
                                            ssink->Finish());
                VLOG(3)<< "buffer is extracted";
                return buffer_;
                }
            
        
 */