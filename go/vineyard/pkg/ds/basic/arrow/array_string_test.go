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

package arrow

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

func TestStringArray(t *testing.T) {
	client, err := client.NewIPCClient(client.GetDefaultIPCSocket())
	if err != nil {
		t.Fatalf("connect to ipc server failed: %+v", err)
	}
	defer client.Disconnect()

	const N uint64 = 1024
	var i uint64

	// create array
	var objectId types.ObjectID
	{
		array := NewStringBuilder()
		if err != nil {
			t.Fatalf("create array failed: %+v", err)
		}
		for i = 0; i < N; i++ {
			array.Append(strconv.Itoa(int(i)))
		}
		objectId, err = array.Seal(client)
		if err != nil {
			t.Fatalf("seal array failed: %+v", err)
		}
	}

	// get array
	var array StringArray
	{
		err := client.GetObject(objectId, &array)
		if err != nil {
			t.Fatalf("get array failed: %+v", err)
		}
	}

	// validate array content
	{
		for i = 0; i < N; i++ {
			if array.Value(int(i)) != strconv.Itoa(int(i)) {
				t.Fatalf("array content not match % d", i)
			}
		}
	}
}

func TestLargeStringArray(t *testing.T) {
	client, err := client.NewIPCClient(client.GetDefaultIPCSocket())
	if err != nil {
		t.Fatalf("connect to ipc server failed: %+v", err)
	}
	defer client.Disconnect()

	const N uint64 = 1024
	var i uint64

	// create array
	var objectId types.ObjectID
	{
		array := NewLargeStringBuilder()
		if err != nil {
			t.Fatalf("create array failed: %+v", err)
		}
		for i = 0; i < N; i++ {
			array.Append(strconv.Itoa(int(i)))
		}
		objectId, err = array.Seal(client)
		if err != nil {
			t.Fatalf("seal array failed: %+v", err)
		}
	}

	// get array
	var array LargeStringArray
	{
		err := client.GetObject(objectId, &array)
		if err != nil {
			t.Fatalf("get array failed: %+v", err)
		}
	}

	// validate array content
	{
		for i = 0; i < N; i++ {
			assert.Equal(t, array.Value(int(i)), strconv.Itoa(int(i)), "array content not match")
		}
	}
}
