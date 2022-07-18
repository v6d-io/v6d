#include "i_adaptor.h"

namespace vineyard {
namespace fuse {
class Mock : public FuseAdaptorRegistered<Mock> {
  public:
  static void* deserialize(void* k){
                        std::clog<<"deserialized is sucessfully called"<<std::endl; 
                return nullptr;
                };
};
}  // namespace fuse

}  // namespace vineyard