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

package client

import (
	"testing"
)

func TestIPCClientConnect(t *testing.T) {
	client, err := NewIPCClient(GetDefaultIPCSocket())
	if err != nil {
		t.Fatalf("connect to ipc server failed: %+v", err)
	}
	defer client.Disconnect()
}

func TestIPCClientBuffer(t *testing.T) {
	client, err := NewIPCClient(GetDefaultIPCSocket())
	if err != nil {
		t.Fatalf("connect to ipc server failed: %+v", err)
	}
	defer client.Disconnect()

	// create buffer
	buffer, err := client.CreateBuffer(1024)
	if err != nil {
		t.Fatalf("create buffer failed: %+v", err)
	}
	bufferBytes := buffer.Bytes()
	for i := 0; i < 1024; i++ {
		bufferBytes[i] = byte(i)
	}

	// get buffer
	buffer2, err := client.GetBuffer(buffer.Id, true)
	if err != nil {
		t.Fatalf("get buffer failed: %+v", err)
	}
	for i := 0; i < 1024; i++ {
		if bufferBytes[i] != buffer2.Bytes()[i] {
			t.Fatalf("buffer content not match")
		}
	}

	// seal buffer
	_, err = buffer.Seal(client)
	if err != nil {
		t.Fatalf("seal buffer failed: %+v", err)
	}

	// get buffer using get object
	var buffer3 Blob
	err = client.GetObject(buffer.Id, &buffer3)
	if err != nil {
		t.Fatalf("get buffer failed: %+v", err)
	}
	for i := 0; i < 1024; i++ {
		if bufferBytes[i] != buffer3.Bytes()[i] {
			t.Fatalf("buffer content not match")
		}
	}

	// create a new buffer
	newBuffer, err := client.CreateBuffer(1024)
	if err != nil {
		t.Fatalf("create newBuffer failed: %+v", err)
	}
	_, err = newBuffer.Seal(client)
	if err != nil {
		t.Fatalf("seal newBuffer failed: %+v", err)
	}

	// list metadatas
	metadatas := map[string]map[string]any{}
	pattern := "*"
	limit := 100
	regex := false
	metadatas, err = client.ListMetadata(pattern, regex, limit)
	if err != nil {
		t.Fatalf("list metadatas failed: %+v", err)
	}

	if len(metadatas) != 2 {
		t.Fatalf("list metadatas failed: %+v", err)
	}

}
