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

import "testing"

func TestRPCServer_Connect(t *testing.T) {
	ipcAddr := "0.0.0.0:9600"
	var rpcServer RPCClient
	err := rpcServer.Connect(ipcAddr)
	if err != nil {
		t.Error("conect to rpc server failed", err.Error())
	}
	err = rpcServer.Disconnect()
	if err != nil {
		t.Error("discconect rpc server failed", err.Error())
	}
}
