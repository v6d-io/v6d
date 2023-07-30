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
package flags

import (
	"testing"

	"github.com/spf13/cobra"
)

/*func TestApplySidecarOpts(t *testing.T) {
	type args struct {
		cmd *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplySidecarOpts(tt.args.cmd)
		})
	}
}*/

func TestApplySidecarOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplySidecarOpts(cmd)

	t.Run("DefaultOptions", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认选项
		if SidecarName != "vineyard-sidecar" {
			t.Errorf("Default sidecar name is incorrect, got: %s, want: 'vineyard-sidecar'", SidecarName)
		}
		if SidecarOpts.Replicas != 1 {
			t.Errorf("Default etcd replicas is incorrect, got: %d, want: 1", SidecarOpts.Replicas)
		}
		if WorkloadYaml != "" {
			t.Errorf("Default workload yaml is incorrect, got: %s, want: ''", WorkloadYaml)
		}
		if WorkloadResource != "" {
			t.Errorf("Default workload resource is incorrect, got: %s, want: ''", WorkloadResource)
		}
		if OwnerReference != "" {
			t.Errorf("Default owner reference is incorrect, got: %s, want: ''", OwnerReference)
		}
		if ApplyResources != false {
			t.Errorf("Default apply resources is incorrect, got: %v, want: false", ApplyResources)
		}
		if OutputFormat != "yaml" {
			t.Errorf("Default output format is incorrect, got: %s, want: 'yaml'", OutputFormat)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

}
