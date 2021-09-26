package ds

import (
	"fmt"
	"github.com/apache/arrow/go/arrow/array"
)

type ObjectBuilder struct {
	sealed bool
}

type ArrayBaseBuilder struct {
	ObjectBuilder
}

func (o *ObjectBuilder) SetSeal(seal bool) {
	o.sealed = seal
}

type ArrayBuilder struct {
	ArrayBaseBuilder

	Array  *array.FixedSizeList
	Client IIPCClient

	length     int
	nullCount  int
	offset     int
	buffer     []byte
	nullBitmap []byte
}

func (a *ArrayBuilder) Init(client IIPCClient, array *array.FixedSizeList) {
	a.Client = client
	a.Array = array
}

func (a *ArrayBuilder) Seal() {
	if a.sealed {
		fmt.Println("The builder has been already been sealed")
		return
	}
	a.Build()

	a.SetSeal(true)
}

func (a *ArrayBuilder) Build() error {
	var blobWriter BlobWriter
	//blobWriter
	a.Client.CreateBlob(a.Array.Data().Len(), &blobWriter)

	a.length = a.Array.Len()
	a.nullCount = a.Array.NullN()
	a.offset = a.Array.Offset()
	a.buffer = blobWriter.Buf()
	// TODO: build null bit map
	return nil
}
