package memory

import (
	"unsafe"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

func Slice(s []byte, offset, length uint64) []byte {
	return s[offset : offset+length]
}

func Cast[T types.Number](s []byte, length uint64) []T {
	return unsafe.Slice((*T)(unsafe.Pointer(&s[0])), length)
}

func CastFrom[T types.Number](pointer unsafe.Pointer, length uint64) []T {
	return unsafe.Slice((*T)(pointer), length)
}
