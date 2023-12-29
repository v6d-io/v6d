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
package inject

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

var etcdResources = `apiVersion: v1
kind: Pod
metadata:
  labels:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: etcd
    etcd_node: vineyard-sidecar-etcd-0
  name: vineyard-sidecar-etcd-0
  namespace: vineyard-system
  ownerReferences: []
spec:
  containers:
  - command:
    - etcd
    - --name
    - vineyard-sidecar-etcd-0
    - --initial-advertise-peer-urls
    - http://vineyard-sidecar-etcd-0:2380
    - --advertise-client-urls
    - http://vineyard-sidecar-etcd-0:2379
    - --listen-peer-urls
    - http://0.0.0.0:2380
    - --listen-client-urls
    - http://0.0.0.0:2379
    - --initial-cluster
    - vineyard-sidecar-etcd-0=http://vineyard-sidecar-etcd-0:2380
    - --initial-cluster-state
    - new
    image: vineyardcloudnative/vineyardd:latest
    imagePullPolicy: IfNotPresent
    name: etcd
    ports:
    - containerPort: 2379
      name: client
      protocol: TCP
    - containerPort: 2380
      name: server
      protocol: TCP
  restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    etcd_node: vineyard-sidecar-etcd-0
  name: vineyard-sidecar-etcd-0
  namespace: vineyard-system
  ownerReferences: []
spec:
  ports:
  - name: client
    port: 2379
    protocol: TCP
    targetPort: 2379
  - name: server
    port: 2380
    protocol: TCP
    targetPort: 2380
  selector:
    app.vineyard.io/role: etcd
    etcd_node: vineyard-sidecar-etcd-0
---
apiVersion: v1
kind: Service
metadata:
  name: vineyard-sidecar-etcd-service
  namespace: vineyard-system
  ownerReferences: []
spec:
  ports:
  - name: etcd-for-vineyard-port
    port: 2379
    protocol: TCP
    targetPort: 2379
  selector:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: etcd
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app.vineyard.io/name: vineyard-sidecar
  name: vineyard-sidecar-rpc
  namespace: vineyard-system
  ownerReferences: []
spec:
  ports:
  - name: vineyard-rpc
    port: 9600
    protocol: TCP
  selector:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: vineyardd
  type: ClusterIP
---
`

func TestInjectCmd(t *testing.T) {
	// set the flags
	flags.Namespace = "vineyard-system"
	flags.KubeConfig = os.Getenv("KUBECONFIG")
	flags.SidecarOpts.Vineyard.Image = "vineyardcloudnative/vineyardd:latest"
	workloadResource := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
		  "name": "nginx-deployment",
		  "namespace": "vineyard-system"
		},
		"spec": {
		  "selector": {
			"matchLabels": {
			  "app": "nginx"
			}
		  },
		  "template": {
			"metadata": {
			  "labels": {
				"app": "nginx"
			  }
			},
			"spec": {
			  "containers": [
				{
				  "name": "nginx",
				  "image": "nginx:1.14.2",
				  "ports": [
					{
					  "containerPort": 80
					}
				  ]
				}
			  ]
			}
		  }
		}
	}`
	podResource := `{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
		  "name": "python",
		  "namespace": "vineyard-system"
		},
		"spec": {
		  "containers": [
			{
			  "name": "python",
			  "image": "python:3.10",
			  "command": [
				"python",
				"-c",
				"import time; time.sleep(100000)"
			  ]
			}
		  ]
		}
	  }`
	//nolint: lll
	injectedWorkload := `apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  name: nginx-deployment
  namespace: vineyard-system
  ownerReferences: []
spec:
  selector:
    matchLabels:
      app: nginx
  strategy: {}
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: nginx
        app.vineyard.io/name: vineyard-sidecar
        app.vineyard.io/role: vineyardd
    spec:
      containers:
      - command: null
        env:
        - name: VINEYARD_IPC_SOCKET
          value: /var/run/vineyard.sock
        image: nginx:1.14.2
        name: nginx
        ports:
        - containerPort: 80
        resources: {}
        volumeMounts:
        - mountPath: /var/run
          name: vineyard-socket
      - command:
        - /bin/bash
        - -c
        - |
          /usr/local/bin/vineyardd --sync_crds true --socket /var/run/vineyard.sock --size  --stream_threshold 80 --etcd_cmd etcd --etcd_prefix /vineyard --etcd_endpoint http://vineyard-sidecar-etcd-service:2379
        env:
        - name: VINEYARDD_UID
          value: null
        - name: VINEYARDD_NAME
          value: vineyard-sidecar
        - name: VINEYARDD_NAMESPACE
          value: vineyard-system
        image: vineyardcloudnative/vineyardd:latest
        imagePullPolicy: IfNotPresent
        name: vineyard-sidecar
        ports:
        - containerPort: 9600
          name: vineyard-rpc
          protocol: TCP
        resources:
          limits: null
          requests: null
        securityContext: {}
        volumeMounts:
        - mountPath: /var/run
          name: vineyard-socket
      volumes:
      - emptyDir: {}
        name: vineyard-socket
status: {}

`
	//nolint: lll
	injectedPod := `apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: vineyardd
  name: python
  namespace: vineyard-system
  ownerReferences: []
spec:
  containers:
  - command:
    - python
    - -c
    - while [ ! -e /var/run/vineyard.sock ]; do sleep 1; done;import time; time.sleep(100000)
    env:
    - name: VINEYARD_IPC_SOCKET
      value: /var/run/vineyard.sock
    image: python:3.10
    name: python
    resources: {}
    volumeMounts:
    - mountPath: /var/run
      name: vineyard-socket
  - command:
    - /bin/bash
    - -c
    - |
      /usr/local/bin/vineyardd --sync_crds true --socket /var/run/vineyard.sock --size  --stream_threshold 80 --etcd_cmd etcd --etcd_prefix /vineyard --etcd_endpoint http://vineyard-sidecar-etcd-service:2379
    env:
    - name: VINEYARDD_UID
      value: null
    - name: VINEYARDD_NAME
      value: vineyard-sidecar
    - name: VINEYARDD_NAMESPACE
      value: vineyard-system
    image: vineyardcloudnative/vineyardd:latest
    imagePullPolicy: IfNotPresent
    name: vineyard-sidecar
    ports:
    - containerPort: 9600
      name: vineyard-rpc
      protocol: TCP
    resources:
      limits: null
      requests: null
    securityContext: {}
    volumeMounts:
    - mountPath: /var/run
      name: vineyard-socket
  volumes:
  - emptyDir: {}
    name: vineyard-socket
status: {}

`
	flags.VineyarddOpts.EtcdReplicas = 1
	//flags.OutputFormat = "json"
	flags.ApplyResources = false
	tests := []struct {
		name     string
		resource string
		wanted   string
	}{
		{
			name:     "inject kubernetes workload resource",
			resource: workloadResource,
			wanted:   etcdResources + injectedWorkload,
		},
		{
			name:     "inject kubernetes pod resource",
			resource: podResource,
			wanted:   etcdResources + injectedPod,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.WorkloadResource = tt.resource
			output := util.CaptureCmdOutput(injectCmd)
			if !reflect.DeepEqual(output, tt.wanted) {
				t.Errorf("%v error: InjectCmd() = %v, want %v", tt.name, output, tt.wanted)
			}
		})
	}
}

func TestValidateFormat(t *testing.T) {
	type args struct {
		format string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name:    "Valid YAML format",
			args:    args{format: YAMLFormat},
			wantErr: false,
		},
		{
			name:    "Valid JSON format",
			args:    args{format: JSONFormat},
			wantErr: false,
		},
		{
			name:    "Invalid format",
			args:    args{format: "invalid"},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateFormat(tt.args.format); (err != nil) != tt.wantErr {
				t.Errorf("validateFormat() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetWorkloadResource(t *testing.T) {
	// get relative path
	workload_YAML := "../../../config/samples/k8s_v1alpha1_sidecar.yaml"

	tests := []struct {
		name           string
		workloadYAML   string
		workloadJSON   string
		expectedResult string
		expectedError  error
	}{
		{
			name:         "Valid YAML file",
			workloadYAML: workload_YAML,
			workloadJSON: "",
			expectedResult: `apiVersion: k8s.v6d.io/v1alpha1
kind: Sidecar
metadata:
  name: sidecar-sample1
  namespace: vineyard-job
spec:
  replicas: 2
  selector: app=sidecar-job-deployment
  vineyard:
    socket: /var/run/vineyard.sock`,
			expectedError: nil,
		},
		{
			name:         "Valid workload resource",
			workloadYAML: "",
			workloadJSON: `{"apiVersion":"apps/v1","kind":"Deployment"}`,
			expectedResult: `apiVersion: apps/v1
kind: Deployment` + "\n",
			expectedError: nil,
		},
		{
			name:           "Both workload yaml and workload resource specified",
			workloadYAML:   workload_YAML,
			workloadJSON:   `{"apiVersion":"apps/v1","kind":"Deployment"}`,
			expectedResult: "",
			expectedError:  errors.New("cannot specify both workload resource and workload yaml"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.WorkloadYaml = tt.workloadYAML
			flags.WorkloadResource = tt.workloadJSON
			result, err := getWorkloadResource()

			if tt.expectedError != nil {
				if err == nil {
					t.Errorf("Expected error %v, but got no error", tt.expectedError)
				} else if err.Error() != tt.expectedError.Error() {
					t.Errorf("Expected error %v, but got %v", tt.expectedError, err)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if result != tt.expectedResult {
					t.Errorf("Expected result:\n%s\n\nBut got:\n%s", tt.expectedResult, result)

				}
			}
		})
	}
}

func TestGetWorkloadObj(t *testing.T) {
	type args struct {
		workload string
	}
	tests := []struct {
		name    string
		args    args
		want    *unstructured.Unstructured
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test case",
			args: args{
				workload: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
spec:
  template:
  spec:
    containers:
    - name: test-container
      image: nginx:latest`,
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "test-deployment",
					},
					"spec": map[string]interface{}{
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
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetWorkloadObj(tt.args.workload)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetWorkloadObj() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetWorkloadObj() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_buildSidecar(t *testing.T) {
	type args struct {
		namespace string
	}
	tests := []struct {
		name    string
		args    args
		want    *v1alpha1.Sidecar
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			args: args{
				namespace: "vineyard-system",
			},
			want: &v1alpha1.Sidecar{
				TypeMeta: metav1.TypeMeta{Kind: "", APIVersion: ""},
				ObjectMeta: metav1.ObjectMeta{
					Name:                       "vineyard-sidecar",
					GenerateName:               "",
					Namespace:                  "vineyard-system",
					SelfLink:                   "",
					UID:                        "",
					ResourceVersion:            "",
					Generation:                 0,
					CreationTimestamp:          metav1.Time{},
					DeletionTimestamp:          nil,
					DeletionGracePeriodSeconds: nil,
					Labels:                     nil,
					Annotations:                nil,
					OwnerReferences:            []metav1.OwnerReference{},
					Finalizers:                 []string{},
					ManagedFields:              []metav1.ManagedFieldsEntry{},
				},
				Spec: v1alpha1.SidecarSpec{
					Selector:     "",
					Replicas:     1,
					EtcdReplicas: 0,
					Vineyard: v1alpha1.VineyardConfig{
						Image:           "vineyardcloudnative/vineyardd:latest",
						ImagePullPolicy: "IfNotPresent",
						SyncCRDs:        true,
						Socket:          "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}",
						ReserveMemory:   false,
						StreamThreshold: 80,
						Spill: v1alpha1.SpillConfig{
							Name:                      "",
							Path:                      "",
							SpillLowerRate:            "0.3",
							SpillUpperRate:            "0.8",
							PersistentVolumeSpec:      corev1.PersistentVolumeSpec{},
							PersistentVolumeClaimSpec: corev1.PersistentVolumeClaimSpec{},
						},
						Env:    []corev1.EnvVar{},
						Memory: "",
						CPU:    "",
					},
					Metric: v1alpha1.MetricConfig{
						Enable:          false,
						Image:           "vineyardcloudnative/vineyard-grok-exporter:latest",
						ImagePullPolicy: "IfNotPresent",
					},
					Volume: v1alpha1.VolumeConfig{},
					Service: v1alpha1.ServiceConfig{
						Type: "ClusterIP",
						Port: 9600,
					},
				},
				Status: v1alpha1.SidecarStatus{
					Current: 0,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := buildSidecar(tt.args.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("buildSidecar() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			gotStr := fmt.Sprintf("%v", got)
			wantedStr := fmt.Sprintf("%v", tt.want)
			if gotStr != wantedStr {
				t.Errorf("buildSidecar() = %#v, want %#v", gotStr, wantedStr)
			}
		})
	}
}
