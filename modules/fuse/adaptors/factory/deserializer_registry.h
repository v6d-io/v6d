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
using vineyard_deserializer_nt =  std::shared_ptr<arrow::Buffer>  (*) (void*);


        std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt> d_registry; 
        template<typename T>
        std::shared_ptr<arrow::Buffer> arrow_ipc_view(
            void* p) {
            auto arr = (T*) p;
            std::clog<<"new registry way"<<std::endl;
            return nullptr;
        //     VLOG(2)<< "arrow_ipc_view is called";
        //     std::shared_ptr<arrow::io::BufferOutputStream> ssink;

        // CHECK_ARROW_ERROR_AND_ASSIGN( ssink, arrow::io::BufferOutputStream::Create(
        //                     ));
        //     VLOG(2)<<"buffer successfully created";

        // auto kvmeta = std::shared_ptr<arrow::KeyValueMetadata>(
        //     new arrow::KeyValueMetadata({}, {}));
            
        // auto meta = arr->meta();
        
        // for (auto i : meta) {
        //     std::string v = i.value().dump();
        //     kvmeta->Append(i.key(), v);
        // }
        // auto schema = arrow::schema(
        //     {arrow::field("a", ConvertToArrowType<int64_t>::TypeValue())}, kvmeta);
        // std::shared_ptr<arrow::Table> my_table =
        //     arrow::Table::Make(schema, {arr->GetArray()});
            
        // std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
        // CHECK_ARROW_ERROR_AND_ASSIGN( writer,
        //                             arrow::ipc::MakeStreamWriter(ssink, schema));
        
        // VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
        // VLOG(4)<< "table is written";
        // writer->Close();
        //     std::shared_ptr<arrow::Buffer> buffer_;
        // VLOG(3)<< "writer is closed";
        // CHECK_ARROW_ERROR_AND_ASSIGN( buffer_,
        //                             ssink->Finish());
        // VLOG(3)<< "buffer is extracted";
        // return buffer_;
        }



        void register_once(){
            d_registry.emplace(type_name<vineyard::NumericArray<int64_t>>(),&arrow_ipc_view<vineyard::NumericArray<int64_t>>);
        };
    } // namespace fuse
    
} // namespace vineyard
