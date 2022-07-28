#include "arrow/array.h"
#include "arrow/buffer.h"



namespace vineyard
{
    namespace fuse
    {
        template<typename T>
        class NumericArrayAdaptor
        {
        public:
            arrow::Buffer Deserialize(std::shared_ptr<vineyard::NumericArray<int64_t>>& arr);
        };
                
    } // namespace fuse
    
    
} // namespace vineyard

