package main

/*
#cgo CXXFLAGS: -std=c++14
#cgo CFLAGS: -I ../../../src/
#include "common/memory/fling.h"
*/
import "C"

func main() {
	C.send_fd(0)
}
