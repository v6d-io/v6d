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

package schedule

import (
	"testing"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

/*func TestScheduleWorkflowCmd(t *testing.T) {
	tests := []struct {
		name string
		want *cobra.Command
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: scheduleWorkflowCmd, // 指定预期的 *cobra.Command 值
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewScheduleWorkflowCmd(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewScheduleWorkflowCmd() = %v, want %v", got, tt.want)
			}
		})
	}
}*/

func TestSchedulingWorkflow(t *testing.T) {
	flags.KubeConfig = "/home/zhuyi/.kube/config"
	flags.WorkflowFile = "/home/zhuyi/v6d/k8s/test/e2e/scheduling-outside-cluster-demo/test.yaml"
	c := util.KubernetesClient()
	manifests, _ := util.ReadFromFile(flags.WorkflowFile)
	objs, _ := util.ParseManifestsToObjects([]byte(manifests))

	type args struct {
		c   client.Client
		obj *unstructured.Unstructured
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "test case",
			args: args{
				c:   c,
				obj: objs[0],
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, obj := range objs {
				if err := SchedulingWorkflow(c, obj); err != nil {
					t.Errorf("SchedulingWorkflow() error = %v", err)
				}
			}
			/*if err := SchedulingWorkflow(tt.args.c, tt.args.obj); (err != nil) != tt.wantErr {
				t.Errorf("SchedulingWorkflow() error = %v, wantErr %v", err, tt.wantErr)
			}*/
		})
	}
}
