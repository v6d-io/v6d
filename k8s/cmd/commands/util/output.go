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
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/pkg/errors"
	"github.com/v6d-io/v6d/go/vineyard/pkg/client"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/types"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	// MetaHeaders are the headers for the less metadata output
	MetaHeaders []string = []string{"ID", "TYPENAME", "LENGTH", "TRANSIENT", "TYPE", "MEMBERS", "INSTANCE ID"}

	// ClusterInfoHeaders are the headers for the cluster info output
	ClusterInfoHeaders []string = []string{"INSTANCE ID", "HOSTID", "HOSTNAME", "IPC_SOCKET", "NODENAME", "RPC_ENDPOINT", "TIMESTAMP"}

	// BufferHeaders are the headers for the buffer output
	BufferHeaders []string = []string{"ID", "BUFFER"}

	// ValidSortedKeys are the valid keys for sorting
	ValidSortedKeys []string = []string{"id", "typename", "type", "instance_id"}

	// ValidOutputFormats are the valid output formats
	ValidOutputFormats []string = []string{"table", "json"}
)

// Options are the options for the output
type Options struct {
	// WithFilter means whether to filter the signature object
	withFilter bool

	// WithSort means whether to sort the output
	sortedKey string

	// format means the output format
	format string
}

// Output is the output of the metadata and buffers
type Output struct {
	Options
	metadatas   *map[string]map[string]any
	buffers     *map[types.ObjectID]client.Blob
	clusterInfo *map[string]any
	order       *[]KeyValue
}

func (o *Output) SortedKey(key string) *Output {
	o.Options.sortedKey = key
	return o
}

func (o *Output) WithFilter(filter bool) *Output {
	o.Options.withFilter = filter
	return o
}

func (o *Output) SetFormat(format string) *Output {
	o.Options.format = format
	return o
}

func NewOutput(metadatas *map[string]map[string]any,
	buffers *map[types.ObjectID]client.Blob,
	clusterInfo *map[string]any) *Output {
	return &Output{
		Options:     Options{},
		metadatas:   metadatas,
		buffers:     buffers,
		clusterInfo: clusterInfo,
	}
}

type KeyValue struct {
	ObjectID string
	Value    string
}

// Filter will filter the signature object
func (o *Output) Filter() *Output {
	if !o.Options.withFilter {
		return o
	}

	if o.metadatas == nil {
		return o
	}

	for i := range *o.metadatas {
		if strings.Contains(i, "s") {
			delete(*o.metadatas, i)
		}
	}
	return o
}

// Sort will sort the output by the specified field
func (o *Output) SortBy() *Output {
	sortedKey := o.Options.sortedKey
	if o.metadatas == nil {
		return o
	}
	// check if the sorted key is valid
	valid := false
	for _, v := range ValidSortedKeys {
		if v == sortedKey {
			valid = true
			break
		}
	}
	if !valid {
		log.Fatal(errors.Errorf("invalid sorted key: %s", sortedKey), "failed to sort the output")
	}
	data := []KeyValue{}
	for k, v := range *o.metadatas {
		if sortedKey == "instance_id" {
			data = append(data, KeyValue{k, v[sortedKey].(json.Number).String()})
		} else if sortedKey == "type" {
			if v["global"] != nil && v["global"].(bool) {
				data = append(data, KeyValue{k, "global"})
			} else {
				data = append(data, KeyValue{k, "local"})
			}
		} else {
			data = append(data, KeyValue{k, v[sortedKey].(string)})
		}
	}
	// sort the data
	sort.Slice(data, func(i, j int) bool {
		return data[i].Value < data[j].Value
	})
	o.order = &data
	return o
}

func (o *Output) Format() {
	format := o.Options.format
	valid := false
	for _, v := range ValidOutputFormats {
		if v == format {
			valid = true
			break
		}
	}
	if !valid {
		log.Fatal(errors.Errorf("invalid output format: %s", format), "failed to format the output")
	}
	if format == "table" {
		o.formatAsTable()
	} else {
		o.formatAsJson()
	}
}

func (o *Output) Print() {
	o.Filter().SortBy().Format()
}

// formatAsJson will format the data as json
func (o *Output) formatAsJson() {
	// print metadatas as json
	if o.metadatas != nil {
		jsonMeta, err := json.MarshalIndent(o.metadatas, "", "  ")
		if err != nil {
			log.Fatal(err, "failed to marshal metadata")
		}
		log.Output(string(jsonMeta))
	}

	type Object struct {
		ID   string `json:"id"`
		Data string `json:"data"`
	}

	if o.buffers != nil {
		objects := make([]Object, 0)
		for id, blob := range *o.buffers {
			d, err := blob.Data()
			if err != nil {
				log.Fatal(err, "failed to get the data of the buffer")
			}
			obj := Object{
				ID:   types.ObjectIDToString(id),
				Data: fmt.Sprintf("%v", d),
			}
			objects = append(objects, obj)
		}
		jsonBuf, err := json.MarshalIndent(objects, "", "  ")
		if err != nil {
			log.Fatal(err, "failed to marshal buffers")
		}
		log.Output(string(jsonBuf))
	}

	if o.clusterInfo != nil {
		jsonClusterInfo, err := json.MarshalIndent(o.clusterInfo, "", "  ")
		if err != nil {
			log.Fatal(err, "failed to marshal cluster info")
		}
		log.Output(string(jsonClusterInfo))
	}
}

// formatAsTable will format the data as table
func (o *Output) formatAsTable() {
	if o.metadatas != nil {
		metaTable := tablewriter.NewWriter(os.Stdout)

		metaTable.SetHeader(MetaHeaders)
		for _, v := range *o.order {
			// set the default value
			var typename, length, transient, members, instanceID string
			objtype := "local"
			// get the value from the map
			item := (*o.metadatas)[v.ObjectID]
			if item["length"] != nil {
				length = item["length"].(json.Number).String()
			}
			if item["transient"] != nil {
				transient = strconv.FormatBool(item["transient"].(bool))
			}
			if item["instance_id"] != nil {
				instanceID = item["instance_id"].(json.Number).String()
			}
			if item["typename"] != nil {
				typename = item["typename"].(string)
			}
			if item["global"] != nil {
				// convert to bool
				isGlobal := item["global"].(bool)
				if isGlobal {
					objtype = "global"
					size, err := item["__elements_-size"].(json.Number).Int64()
					if err != nil {
						log.Fatal(err, "failed to get the size of the global object")
					}
					allMembers := []string{}
					for i := 0; i < int(size); i++ {
						allMembers = append(allMembers, item["__elements_-"+strconv.Itoa(i)].(map[string]any)["id"].(string))
						members = strings.Join(allMembers, "\n")
					}
				}
			}
			metaTable.Append([]string{v.ObjectID, typename, length, transient, objtype, members, instanceID})
		}
		metaTable.Render()
	}

	// Write buffer table to output if there are any buffers
	if o.buffers != nil {
		bufferTable := tablewriter.NewWriter(os.Stdout)
		bufferTable.SetHeader(BufferHeaders)
		for id, blob := range *o.buffers {
			data, err := blob.Data()
			if err != nil {
				log.Fatal(err, "failed to get the data of the buffer")
			}
			bufferTable.Append([]string{types.ObjectIDToString(id), fmt.Sprintf("%v", data)})
		}
		bufferTable.Render()
	}

	if o.clusterInfo != nil {
		clusterInfoTable := tablewriter.NewWriter(os.Stdout)

		clusterInfoTable.SetHeader(ClusterInfoHeaders)
		for instance, info := range *o.clusterInfo {
			// set the default value
			var hostid, hostname, ipcSocket, nodename, rpcEndpoint, timeStamp string
			hostid = info.(map[string]any)["hostid"].(json.Number).String()
			hostname = info.(map[string]any)["hostname"].(string)
			ipcSocket = info.(map[string]any)["ipc_socket"].(string)
			nodename = info.(map[string]any)["nodename"].(string)
			rpcEndpoint = info.(map[string]any)["rpc_endpoint"].(string)
			timeStamp = info.(map[string]any)["timestamp"].(json.Number).String()
			clusterInfoTable.Append([]string{instance, hostid, hostname, ipcSocket, nodename, rpcEndpoint, timeStamp})
		}
		clusterInfoTable.Render()
	}
}
