/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vineyard

import (
	"fmt"
	"testing"

	"github.com/apache/arrow/go/arrow"
	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"
	vineyard "github.com/v6d-io/v6d/go/vineyard/pkg/client/ds"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
)

func TestIPCServer_Connect(t *testing.T) {
	ipcAddr := "/var/run/vineyard.sock"
	ipcServer := IPCClient{}
	err := ipcServer.Connect(ipcAddr)
	if err != nil {
		t.Error("connect to ipc server failed", err)
	}
	err = ipcServer.Disconnect()
	if err != nil {
		t.Error("disconnect ipc server failed", err.Error())
	}
}

func TestIPCClient_GetName(t *testing.T) {
	ipcAddr := "/var/run/vineyard.sock"
	name := "test_name"
	nameNoExist := "undefined_name"
	ipcServer := IPCClient{}
	err := ipcServer.Connect(ipcAddr)
	if err != nil {
		t.Error("connect to ipc server failed", err)
	}
	var id1 common.ObjectID = common.GenerateObjectID()
	if err := ipcServer.PutName(id1, name); err != nil {
		if putErr, ok := err.(*common.ReplyError); ok {
			t.Log("get name return code", putErr.Code)
		} else {
			t.Error("get name failed", err)
		}
	}
	var id2 common.ObjectID
	if err := ipcServer.GetName(name, false, &id2); err != nil {
		if getErr, ok := err.(*common.ReplyError); ok {
			if getErr.Code == common.KObjectNotExists {
				t.Log("get object not exist")
			}
		} else {
			t.Error("get name failed", err)
		}
	}
	if id1 != id2 {
		t.Error("put name id and get name id is not match")
	}
	t.Log("put name and get name success!")

	var id3 common.ObjectID
	if err := ipcServer.GetName(nameNoExist, false, &id3); err != nil {
		if getErr, ok := err.(*common.ReplyError); ok {
			if getErr.Code == common.KObjectNotExists {
				t.Log("get object not exist")
			}
		} else {
			t.Error("get name failed", err)
		}
	}
	t.Log("get no exist name test success")

	if err := ipcServer.DropName(name); err != nil {
		if dropErr, ok := err.(*common.ReplyError); ok {
			if dropErr.Code == common.KObjectNotExists {
				t.Log("drop object not exist")
			}
		} else {
			t.Error("drop name failed", err)
		}
	}
	t.Log("drop name success")

	if err := ipcServer.GetName(name, false, &id1); err != nil {
		if getErr, ok := err.(*common.ReplyError); ok {
			if getErr.Code == common.KObjectNotExists {
				t.Log("get object not exist")
			}
		} else {
			t.Error("get name failed", err)
		}
	}

	err = ipcServer.Disconnect()
	if err != nil {
		t.Error("disconnect ipc server failed", err.Error())
	}
}

// need root privilege to run
// TODO: unfinished
func TestIPCClient_ArrowDataStructure(t *testing.T) {
	pool := memory.NewGoAllocator()

	lb := array.NewFixedSizeListBuilder(pool, 3, arrow.PrimitiveTypes.Int64)
	defer lb.Release()

	vb := lb.ValueBuilder().(*array.Int64Builder)
	defer vb.Release()

	vb.Reserve(10)

	lb.Append(true)
	vb.Append(0)
	vb.Append(1)
	vb.Append(2)

	lb.AppendNull()
	vb.AppendValues([]int64{-1, -1, -1}, nil)

	lb.Append(true)
	vb.Append(3)
	vb.Append(4)
	vb.Append(5)

	lb.Append(true)
	vb.Append(6)
	vb.Append(7)
	vb.Append(8)

	lb.AppendNull()

	arr := lb.NewArray().(*array.FixedSizeList)
	defer arr.Release()

	fmt.Printf("NullN()   = %d\n", arr.NullN())
	fmt.Printf("Len()     = %d\n", arr.Len())
	fmt.Printf("Type()    = %v\n", arr.DataType())
	fmt.Printf("List      = %v\n", arr)

	ipcAddr := "/var/run/vineyard.sock"
	ipcClient := IPCClient{}
	err := ipcClient.Connect(ipcAddr)
	if err != nil {
		t.Error("connect to ipc server failed", err)
	}

	var array vineyard.ArrayBuilder
	array.Init(&ipcClient, arr)
	array.Seal()
	// TODO: how to get array's id

	ipcClient.Persist(array.Id())
}

func TestJSON(t *testing.T) {
	var a map[string]string = make(map[string]string)
	a["a"] = "12"
	if _, ok := a["a"]; ok {
		fmt.Println("has it")
	}
	fmt.Println(a["1"])
}
