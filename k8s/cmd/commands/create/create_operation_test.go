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
package create

import (
	"reflect"
	"testing"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func Test_buildOperation(t *testing.T) {
	tests := []struct {
		name string
		want *v1alpha1.Operation
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Operation{
				ObjectMeta: metav1.ObjectMeta{
					Name:      flags.OperationName,
					Namespace: flags.GetDefaultVineyardNamespace(),
				},
				Spec: flags.OperationOpts,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := buildOperation(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("buildOperation() = %v, want %v", got, tt.want)
			}
		})
	}
}

/*func TestNewCreateOperationCmd(t *testing.T) {
	tests := []struct {
		name string
		want *cobra.Command
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: createOperationCmd, // 指定预期的 *cobra.Command 值
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewCreateOperationCmd(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewCreateOperationCmd() = %v, want %v", got, tt.want)
			}
		})
	}
}*/
