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

package ds

import (
	"strconv"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
)

type ObjectMeta struct {
	client     *IIPCClient
	meta       map[string]interface{}
	bufferSet  BufferSet
	inComplete bool
}

func (o *ObjectMeta) Init() {
	o.meta = make(map[string]interface{})
}

func (o *ObjectMeta) SetClient(client *IIPCClient) {
	o.client = client
}

func (o *ObjectMeta) GetClient() *IIPCClient {
	return o.client
}

func (o *ObjectMeta) SetInstanceId(id common.InstanceID) {
	o.meta["instance_id"] = id
}

func (o *ObjectMeta) AddKeyValue(key string, value interface{}) {
	o.meta[key] = value
}

func (o *ObjectMeta) HasKey(key string) bool {
	if _, ok := o.meta[key]; !ok {
		return false
	}
	return true
}

func (o *ObjectMeta) SetNBytes(nbytes int) {
	o.meta["nbytes"] = nbytes
}

func (o *ObjectMeta) InComplete() bool {
	return o.inComplete
}

func (o *ObjectMeta) MetaData() interface{} {
	return o.meta
}

func (o *ObjectMeta) SetId(id common.ObjectID) {
	o.meta["id"] = strconv.FormatUint(id, 10)
}

func (o *ObjectMeta) SetSignature(signature common.Signature) {
	o.meta["signature"] = signature
}

func (o *ObjectMeta) Reset() {
	o.client = nil
	o.meta = make(map[string]interface{})
	o.bufferSet.Reset()
	o.inComplete = false
}

func (o *ObjectMeta) SetMetaData(client *IIPCClient, val map[string]interface{}) {
	o.client = client
	o.meta = val
}
