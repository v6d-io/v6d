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

package schedule

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"sync"
	"testing"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"

	//corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestScheduleWorkloadCmd(t *testing.T) {
	// set the flags
	flags.Namespace = "vineyard-system"
	//flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
	flags.KubeConfig = "/tmp/e2e-k8s.config"
	flags.VineyarddOpts.Replicas = 1
	flags.VineyarddOpts.EtcdReplicas = 1
	flags.Resource = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-container
        image: nginx:latest`

	testReplicas := struct {
		name             string
		vineyardReplicas int
		etcdReplicas     int
	}{
		name:             "test replicas",
		vineyardReplicas: 1,
		etcdReplicas:     1,
	}
	t.Run(testReplicas.name, func(t *testing.T) {
		// Save the original os.Stdout for recovery after testing is completed
		oldStdout := os.Stdout
		var mu sync.Mutex
		mu.Lock() // add a lock

		// Create a pipeline and redirect os.Stdout to the pipeline
		r, w, _ := os.Pipe()
		os.Stdout = w

		// In a Go process, execute your command and then close the write end
		go func() {
			scheduleWorkloadCmd.Run(scheduleWorkloadCmd, []string{})
			w.Close()
		}()

		// In the main Go process, read data from the pipeline
		out, _ := io.ReadAll(r)

		// 'out' now includes the content printed by log.Output
		yamlStr := string(out)
		mu.Unlock()           // unlock
		os.Stdout = oldStdout // redirect os.Stdout to the original one
		want := `{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"creationTimestamp":null,` +
			`"name":"my-deployment"},"spec":{"replicas":1,"selector":null,"strategy":{},"template":{"metadata":` +
			`{"creationTimestamp":null,"labels":{"app":"my-app"}},"spec":{"affinity":{"podAffinity":` +
			`{"requiredDuringSchedulingIgnoredDuringExecution":[{"labelSelector":{"matchExpressions":` +
			`[{"key":"app.kubernetes.io/instance","operator":"In","values":["vineyard-system-vineyardd-sample"]}]},"namespaces":["vineyard-system"],` +
			`"topologyKey":"kubernetes.io/hostname"}]}},"containers":[{"env":[{"name":"VINEYARD_IPC_SOCKET",` +
			`"value":"/var/run/vineyard.sock"}],"image":"nginx:latest","name":"my-container","resources":{},"volumeMounts":` +
			`[{"mountPath":"/var/run","name":"vineyard-socket"}]}],"volumes":[{"hostPath":{"path":` +
			`"/var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample"},"name":"vineyard-socket"}]}}},"status":{}}` + "\n\n"
		if !reflect.DeepEqual(yamlStr, want) {
			t.Errorf("getWorkload() got = %v, want %v", yamlStr, want)
		}
	})
}

func TestValidateWorkloadKind(t *testing.T) {
	type args struct {
		kind string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		// Add test cases.
		{
			name: "Valid Deployment kind",
			args: args{kind: "Deployment"},
			want: true,
		},
		{
			name: "Valid StatefulSet kind",
			args: args{kind: "StatefulSet"},
			want: true,
		},
		{
			name: "Valid ReplicaSet kind",
			args: args{kind: "ReplicaSet"},
			want: true,
		},
		{
			name: "Valid Job kind",
			args: args{kind: "Job"},
			want: true,
		},
		{
			name: "Valid CronJob kind",
			args: args{kind: "CronJob"},
			want: true,
		},
		{
			name: "Invalid kind",
			args: args{kind: "InvalidKind"},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidateWorkloadKind(tt.args.kind); got != tt.want {
				t.Errorf("ValidateWorkloadKind() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_getWorkload(t *testing.T) {
	type args struct {
		workload string
	}
	tests := []struct {
		name    string
		args    args
		want    *unstructured.Unstructured
		want1   bool
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test case",
			args: args{
				workload: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
  template:
  metadata:
    labels:
      app: my-app
  spec:
    containers:
    - name: my-container
      image: nginx:latest`,
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "my-deployment",
					},
					"spec": map[string]interface{}{
						"replicas": 3,
						"selector": nil,
						"strategy": map[string]interface{}{},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"creationTimestamp": nil,
							},
							"spec": map[string]interface{}{
								"containers": nil,
							},
						},
					},
					"status": map[string]interface{}{},
				},
			},
			want1:   true,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := getWorkload(tt.args.workload)
			if (err != nil) != tt.wantErr {
				t.Errorf("getWorkload() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			a := fmt.Sprint(got)
			b := fmt.Sprint(tt.want)
			if !reflect.DeepEqual(a, b) {
				t.Errorf("getWorkload() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("getWorkload() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestSchedulingWorkload(t *testing.T) {
	// Set up test flags
	//flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
	flags.KubeConfig = "/tmp/e2e-k8s.config"
	c := util.KubernetesClient()

	type args struct {
		c               client.Client
		unstructuredObj *unstructured.Unstructured
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test case",
			args: args{
				c: c,
				unstructuredObj: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"kind":       "Deployment",
						"apiVersion": "apps/v1",
						"metadata": map[string]interface{}{
							"name":      "example-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"template": map[string]interface{}{
								"spec": map[string]interface{}{
									"affinity": map[string]interface{}{
										"podAffinity": map[string]interface{}{
											"requiredDuringSchedulingIgnoredDuringExecution": []interface{}{},
										},
									},
								},
							},
						},
					},
				},
			},
			want: `{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"example-deployment",` +
				`"namespace":"default"},"spec":{"template":{"spec":{"affinity":{"podAffinity":` +
				`{"requiredDuringSchedulingIgnoredDuringExecution":[{"labelSelector":` +
				`{"matchExpressions":[{"key":"app.kubernetes.io/instance","operator":"In",` +
				`"values":["vineyard-system-vineyardd-sample"]}]},"topologyKey":` +
				`"kubernetes.io/hostname"}]}},"containers":null,"volumes":[{"hostPath":` +
				`{"path":"/var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample"},"name":"vineyard-socket"}]}}}}` +
				"\n",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := SchedulingWorkload(tt.args.c, tt.args.unstructuredObj)
			if (err != nil) != tt.wantErr {
				t.Errorf("SchedulingWorkload() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			a, _ := got.MarshalJSON()
			if !reflect.DeepEqual(string(a), tt.want) {
				t.Errorf("SchedulingWorkload() = %v, want %v", got, tt.want)
			}
		})
	}
}
