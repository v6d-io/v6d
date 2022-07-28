#include "common/util/typename.h"

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/macros.h"
#include "basic/ds/array.h"
#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/core_types.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"
#include "basic/ds/array.h"
#include<unordered_map>

namespace vineyard
{
    namespace fuse
    {
using vineyard_deserializer_nt =  std::shared_ptr<arrow::Buffer>  (*) (std::shared_ptr<vineyard::Object>);


        std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt> d_array_registry; 
        template<typename T>
        std::shared_ptr<arrow::Buffer> numeric_array_arrow_ipc_view(
            std::shared_ptr<vineyard::Object>p) {
            auto arr = std::dynamic_pointer_cast<T>( p);
            VLOG(2)<< "arrow_ipc_view is called";
            std::shared_ptr<arrow::io::BufferOutputStream> ssink;

        CHECK_ARROW_ERROR_AND_ASSIGN( ssink, arrow::io::BufferOutputStream::Create(
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




        std::shared_ptr<arrow::Buffer> string_array_arrow_ipc_view(
            std::shared_ptr<vineyard::Object> p) {
            auto arr = std::dynamic_pointer_cast<vineyard::BaseBinaryArray<arrow::LargeStringArray>>( p);
            VLOG(2)<< "arrow_ipc_view is called";
            std::shared_ptr<arrow::io::BufferOutputStream> ssink;

        CHECK_ARROW_ERROR_AND_ASSIGN( ssink, arrow::io::BufferOutputStream::Create(
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
            {arrow::field("a", ConvertToArrowType<std::string>::TypeValue())}, kvmeta);
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



        std::shared_ptr<arrow::Buffer> bool_array_arrow_ipc_view(
            std::shared_ptr<vineyard::Object> p) {
            auto arr = std::dynamic_pointer_cast<vineyard::BooleanArray>(p);
            std::clog<<"new registry way"<<std::endl;
            VLOG(2)<< "arrow_ipc_view is called";
            std::shared_ptr<arrow::io::BufferOutputStream> ssink;

        CHECK_ARROW_ERROR_AND_ASSIGN( ssink, arrow::io::BufferOutputStream::Create(
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
            {arrow::field("a", ConvertToArrowType<bool>::TypeValue())}, kvmeta);
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
    

        void register_once(){
              #define FUSE_REGSITER(T)\
                    VLOG(0)<<"register type: " << type_name<T>()<<std::endl;\
                    d_array_registry.emplace(type_name<T>(),&numeric_array_arrow_ipc_view<T>);

            FUSE_REGSITER(vineyard::NumericArray<int8_t>);
            FUSE_REGSITER(vineyard::NumericArray<int32_t>);
            FUSE_REGSITER(vineyard::NumericArray<int16_t>);
            FUSE_REGSITER(vineyard::NumericArray<int64_t>);

            FUSE_REGSITER(vineyard::NumericArray<uint16_t>);
            FUSE_REGSITER(vineyard::NumericArray<uint8_t>);
            FUSE_REGSITER(vineyard::NumericArray<uint32_t>);
            FUSE_REGSITER(vineyard::NumericArray<uint64_t>);
            FUSE_REGSITER(vineyard::NumericArray<float>);
            FUSE_REGSITER(vineyard::NumericArray<double>);
            // d_array_registry.emplace(type_name<vineyard::NumericArray<int64_t>>(),&arrow_ipc_view<vineyard::NumericArray<int64_t>>);
            d_array_registry.emplace(type_name<vineyard::BooleanArray>(),&bool_array_arrow_ipc_view);
            d_array_registry.emplace(type_name<vineyard::BaseBinaryArray<arrow::LargeStringArray>>(), &string_array_arrow_ipc_view);

        };
    } // namespace fuse
    
} // namespace vineyard
