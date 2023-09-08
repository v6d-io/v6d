/*
* Copyright 2020-2023 Alibaba Group Holding Limited.

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
package util

import (
	"github.com/pkg/errors"

	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
)

// Objects are the operate interface for all objects
type Objects interface {
	ListMetadatas(string, bool, int) (map[string]map[string]any, error)
	ListBlobs() (map[types.ObjectID]client.Blob, error)
	GetClusterInfo() (map[string]map[string]string, error)
	GetBlob(id string, unsafe bool) (blob map[types.ObjectID]client.Blob, err error)
	GetMetaData(id string, syncRemote bool) (meta *client.ObjectMeta, err error)
}

// ListMetadatas lists all vineyard metadata
func (c *Client) ListMetadatas(pattern string, regex bool, limits int) (map[string]map[string]any, error) {
	if c.ipcClient != nil {
		return c.ipcClient.ListMetadata(pattern, regex, limits)
	}
	if c.rpcClient != nil {
		return c.rpcClient.ListMetaData(pattern, regex, limits)
	}
	return nil, nil
}

// ListBlobs lists all vineyard blobs
func (c *Client) ListBlobs(limit int) (map[types.ObjectID]client.Blob, error) {
	unsafe := true
	data := make(map[types.ObjectID]client.Blob)
	objs := make([]types.ObjectID, 0)
	metas, err := c.ListMetadatas("vineyard::Blob", false, limit)
	if err != nil {
		return data, err
	}
	for _, m := range metas {
		id, err := types.ObjectIDFromString(m["id"].(string))
		if err != nil {
			return nil, errors.Errorf("failed to parse object id: %s", m["id"].(string))
		}
		objs = append(objs, id)
	}
	if c.ipcClient != nil {
		buffers, err := c.ipcClient.GetBuffers(objs, unsafe)
		if err != nil {
			return nil, err
		}
		return buffers, nil
	}
	return data, errors.Errorf("no ipc client")
}

// GetClusterInfo returns the cluster info
func (c *Client) GetClusterInfo() (map[string]any, error) {
	if c.ipcClient != nil {
		return c.ipcClient.GetClusterInfo()
	}
	if c.rpcClient != nil {
		return c.rpcClient.GetClusterInfo()
	}
	return nil, nil
}

// GetMetadata
func (c *Client) GetMetaData(id string, syncRemote bool) (meta *client.ObjectMeta, err error) {
	object_id, _ := types.ObjectIDFromString(id)
	if c.ipcClient != nil {
		return c.ipcClient.GetMetaData(object_id, syncRemote)
	}
	if c.rpcClient != nil {
		return c.rpcClient.GetMetaData(object_id, syncRemote)
	}
	return nil, nil
}

// GetBlob
func (c *Client) GetBlob(id string, unsafe bool) (blob map[types.ObjectID]client.Blob, err error) {
	meta, err := c.GetMetaData(id, false)
	if err != nil {
		return nil, errors.Errorf("failed to get metadata: %s", err)
	}
	object_id := meta.GetBuffers().GetBufferIds()
	if c.ipcClient != nil {
		Blob, err := c.ipcClient.GetBuffer(object_id[0], unsafe)
		if err != nil {
			return nil, errors.Errorf("failed to get blob: %s", err)
		}
		blob := make(map[types.ObjectID]client.Blob)
		blob[object_id[0]] = Blob
		return blob, nil
	}
	return nil, nil
}

// PutBlob
func (c *Client) PutBlob(address []byte, size uint64) (types.ObjectID, error) {
	if c.ipcClient != nil {
		return c.ipcClient.BuildBuffer(address, size)
	}
	return uint64(0), nil
}
