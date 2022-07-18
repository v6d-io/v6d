#include "i_adaptor.h"

namespace vineyard {
namespace fuse {
class Mock : public FuseAdaptorRegistered<Mock> {
  public:
  static void* deserialize(void* k){
                int* a = new int(2);
                        std::clog<<"deserialized is sucessfully called"<<std::endl; 
                return (void*)a;
                };
};
}  // namespace fuse

}  // namespace vineyard