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

import "github.com/v6d-io/v6d/go/vineyard/pkg/common/types"

type ObjectBase interface {
	Build(client *IPCClient) error
	Seal(client *IPCClient) (types.ObjectID, error)
}

type IObject interface {
	Construct(client *IPCClient, meta *ObjectMeta) error
}

type Object struct {
	Id   types.ObjectID
	Meta *ObjectMeta
}

func (o *Object) Build(client *IPCClient) error {
	return nil
}

func (o *Object) Seal(client *IPCClient) (types.ObjectID, error) {
	return o.Id, nil
}

func (o *Object) NBytes() uint64 {
	if nbytes, err := o.Meta.GetNBytes(); err != nil {
		return 0
	} else {
		return nbytes
	}
}

type IObjectBuilder interface {
	Build(client *IPCClient) error
	Seal(client *IPCClient) (types.ObjectID, error)
}
