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

package basic

import (
	"testing"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

func TestArray(t *testing.T) {
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
		array, err := NewArrayBuilder[int64](client, N)
		if err != nil {
			t.Fatalf("create array failed: %+v", err)
		}
		for i = 0; i < N; i++ {
			array.At(i, int64(i))
		}
		objectId, err = array.Seal(client)
		if err != nil {
			t.Fatalf("seal array failed: %+v", err)
		}
	}

	// get array
	var array Array[int64]
	{
		err := client.GetObject(objectId, &array)
		if err != nil {
			t.Fatalf("get array failed: %+v", err)
		}
	}

	// validate array content
	{
		for i = 0; i < N; i++ {
			if array.At(i) != int64(i) {
				t.Fatalf("array content not match % d", i)
			}
		}
	}
}
