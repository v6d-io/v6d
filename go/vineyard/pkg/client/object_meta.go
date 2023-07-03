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
	arrow "github.com/apache/arrow/go/v11/arrow/memory"
	"github.com/pkg/errors"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

type ObjectMeta struct {
	meta       map[string]any
	client     *ClientBase
	bufferSet  BufferSet
	inComplete bool
}

func NewObjectMeta() *ObjectMeta {
	meta := &ObjectMeta{
		meta: make(map[string]any),
	}
	meta.bufferSet.Reset()
	return meta
}

func (m *ObjectMeta) Init() {
	m.meta = make(map[string]any)
}

func (m *ObjectMeta) GetId() (types.ObjectID, error) {
	if id, err := common.GetString(m.meta, "id"); err == nil {
		return types.ObjectIDFromString(id)
	} else {
		return types.InvalidObjectID(), errors.Wrap(err, "id not found")
	}
}

func (m *ObjectMeta) SetId(id types.ObjectID) {
	m.meta["id"] = types.ObjectIDToString(id)
}

func (m *ObjectMeta) GetSignature() (types.Signature, error) {
	if signature, err := common.GetUint64(m.meta, "signature"); err == nil {
		return signature, nil
	} else {
		return types.InvalidObjectID(), errors.Wrap(err, "signature not found")
	}
}

func (m *ObjectMeta) SetSignature(signature types.Signature) {
	m.meta["signature"] = signature
}

func (m *ObjectMeta) GetInstanceId() (types.InstanceID, error) {
	if id, err := common.GetUint64(m.meta, "instance_id"); err == nil {
		return id, nil
	} else {
		return 0, errors.Wrap(err, "instance_id not found")
	}
}

func (m *ObjectMeta) SetInstanceId(id types.InstanceID) {
	m.meta["instance_id"] = id
}

func (m *ObjectMeta) IsTransient() bool {
	if transient, err := common.GetBoolean(m.meta, "transient"); err == nil {
		return transient
	}
	return true
}

func (m *ObjectMeta) SetTransient(transient bool) {
	m.meta["transient"] = true
}

func (m *ObjectMeta) SetTypeName(typeName string) {
	m.meta["typename"] = typeName
}

func (m *ObjectMeta) GetTypeName() (string, error) {
	if typename, err := common.GetString(m.meta, "typename"); err == nil {
		return typename, nil
	} else {
		return "", errors.Wrap(err, "typename not found")
	}
}

func (m *ObjectMeta) SetGlobal() {
	m.meta["global"] = true
}

func (m *ObjectMeta) IsGlobal() bool {
	if global, err := common.GetBoolean(m.meta, "global"); err == nil {
		return global
	}
	return false
}

func (m *ObjectMeta) IsLocal() bool {
	return !m.IsGlobal()
}

func (m *ObjectMeta) SetNBytes(nbytes uint64) {
	m.meta["nbytes"] = nbytes
}

func (m *ObjectMeta) GetNBytes() (uint64, error) {
	if nbytes, err := common.GetUint64(m.meta, "nbytes"); err == nil {
		return nbytes, nil
	} else {
		return 0, errors.Wrap(err, "nbytes not found")
	}
}

func (m *ObjectMeta) AddKeyValue(key string, value any) {
	m.meta[key] = value
}

func (m *ObjectMeta) GetKeyValue(key string) (any, error) {
	if value, ok := m.meta[key]; ok {
		return value, nil
	}
	return nil, common.Error(common.KMetaTreeInvalid, "key not found")
}

func (m *ObjectMeta) GetKeyValueBool(key string) (bool, error) {
	return common.GetBoolean(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueInt(key string) (int, error) {
	return common.GetInt(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueUint(key string) (uint, error) {
	return common.GetUint(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueInt64(key string) (int64, error) {
	return common.GetInt64(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueUInt64(key string) (uint64, error) {
	return common.GetUint64(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueFloat32(key string) (float32, error) {
	return common.GetFloat32(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueFloat64(key string) (float64, error) {
	return common.GetFloat64(m.meta, key)
}

func (m *ObjectMeta) GetKeyValueString(key string) (string, error) {
	return common.GetString(m.meta, key)
}

func (m *ObjectMeta) AddMember(key string, value *ObjectMeta) {
	m.meta[key] = value.meta
}

func (m *ObjectMeta) AddMemberObject(key string, value *Object) {
	m.meta[key] = value.Meta
}

func (m *ObjectMeta) AddMemberId(key string, value types.ObjectID) {
	node := make(map[string]any)
	node["id"] = types.ObjectIDToString(value)
	m.meta[key] = node
	m.inComplete = true
}

func (m *ObjectMeta) GetMemberMeta(key string) (*ObjectMeta, error) {
	if value, ok := m.meta[key]; ok {
		meta := NewObjectMeta()
		meta.SetMetaData(m.client, value.(map[string]any))
		for _, id := range meta.bufferSet.GetBufferIds() {
			_ = meta.bufferSet.EmplaceBuffer(id, m.bufferSet.buffers[id])
		}
		return meta, nil
	}
	return nil, common.Error(common.KMetaTreeInvalid, "key not found")
}

func (m *ObjectMeta) GetMember(client *IPCClient, key string, object IObject) error {
	if meta, err := m.GetMemberMeta(key); err != nil {
		return nil
	} else {
		return object.Construct(client, meta)
	}
}

func (m *ObjectMeta) HasKey(key string) bool {
	_, ok := m.meta[key]
	return ok
}

func (m *ObjectMeta) InComplete() bool {
	return m.inComplete
}

func (m *ObjectMeta) MetaData() map[string]any {
	return m.meta
}

func (m *ObjectMeta) Reset() {
	m.meta = make(map[string]any)
	m.bufferSet.Reset()
	m.inComplete = false
}

func (m *ObjectMeta) SetMetaData(client *ClientBase, meta map[string]any) {
	m.client = client
	m.meta = meta
	m.traverse(client, meta)
}

func (m *ObjectMeta) GetBuffers() *BufferSet {
	return &m.bufferSet
}

func (m *ObjectMeta) GetBuffer(id types.ObjectID) (*arrow.Buffer, error) {
	return m.bufferSet.Get(id)
}

func (m *ObjectMeta) traverse(client *ClientBase, meta any) {
	if meta == nil {
		return
	}
	switch meta.(type) {
	case map[string]any:
		id, ok := meta.(map[string]any)["id"]
		if !ok {
			return
		}
		memberId, err := types.ObjectIDFromString(id.(string))
		if err != nil {
			return
		}
		if types.IsBlob(memberId) {
			instanceId, err := m.GetInstanceId()
			if err != nil {
				return
			}
			if client != nil && instanceId == client.InstanceID {
				_ = m.bufferSet.EmplaceBufferId(memberId)
			}
		} else {
			for _, v := range meta.(map[string]any) {
				m.traverse(client, v)
			}
		}
	default:
		return
	}
}
