#include "common/util/logging.h"
#include <iostream>
int main() {
  vineyard::logging::InitGoogleLogging("vineyard");

  VLOG(0) << "this is level 0";
  VLOG(2) << "this is level 2";
  VLOG(4) << "this is level 4";
  return 0;
}