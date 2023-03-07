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
	"fmt"
	"testing"
)

func TestRPCClientConnect(t *testing.T) {
	host, port := GetDefaultRPCHostAndPort()
	client, err := NewRPCClient(fmt.Sprintf("%s:%d", host, port))
	if err != nil {
		t.Fatalf("connect to rpc server failed, +%v", err)
	}
	defer client.Disconnect()
}
