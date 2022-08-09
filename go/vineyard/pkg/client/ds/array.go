package ds

import (
	"fmt"

	"github.com/apache/arrow/go/arrow/array"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
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

	Array      *array.FixedSizeList
	Client     IIPCClient
	blobWriter BlobWriter

	length    int
	nullCount int
	offset    int
	buffer    []byte
	//nullBitmap []byte
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
	if err := a.Build(); err != nil {
		fmt.Println("ArrayBuilder Build Failed: ", err)
	}

	a.SetSeal(true)
}

func (a *ArrayBuilder) Build() error {

	//blobWriter
	a.Client.CreateBlob(a.Array.Data().Len(), &a.blobWriter)

	a.length = a.Array.Len()
	a.nullCount = a.Array.NullN()
	a.offset = a.Array.Offset()
	a.buffer = a.blobWriter.Buf()
	// TODO: build null bit map
	return nil
}

func (a *ArrayBuilder) Id() common.ObjectID {
	return a.blobWriter.ID
}
