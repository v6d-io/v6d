package main

/*
#include "test.h"
#include "fling.h"
*/
import "C"
import "fmt"

func main() {
	C.test()
	ret := C.send_fd(0, 0)
	fmt.Println(ret)
}
