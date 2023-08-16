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
package sidecar

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
)

func TestInjectCmd(t *testing.T) {
	// set the flags
	flags.Namespace = "vineyard-system"
	flags.KubeConfig = os.Getenv("KUBECONFIG")
	flags.SidecarOpts.Vineyard.Image = "vineyardcloudnative/vineyardd:alpine-latest"
	flags.WorkloadResource = `{
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
	flags.VineyarddOpts.EtcdReplicas = 1
	flags.ApplyResources = true
	test := struct {
		name                 string
		etcdReplicas         int
		expectedImage        string
		expectedCpu          string
		expectedMemery       string
		expectedService_port int
		expectedService_type string
	}{
		name:                 "test",
		expectedImage:        "vineyardcloudnative/vineyardd:alpine-latest",
		expectedCpu:          "",
		expectedMemery:       "",
		expectedService_port: 9600,
		expectedService_type: "ClusterIP",
	}
	t.Run(test.name, func(t *testing.T) {
		injectCmd.Run(injectCmd, []string{})
		time.Sleep(1 * time.Second)
		// get the replicas of etcd
		k8sclient := util.KubernetesClient()
		etcdPod := corev1.PodList{}
		etcdOpts := []client.ListOption{
			client.InNamespace(flags.Namespace),
			client.MatchingLabels{
				"app.vineyard.io/role": "etcd",
			},
		}
		err := k8sclient.List(context.Background(), &etcdPod, etcdOpts...)
		if err != nil {
			t.Errorf("list etcd pods error: %v", err)
		}

		for _, pod := range etcdPod.Items {
			for _, container := range pod.Spec.Containers {
				if container.Image != test.expectedImage {
					t.Errorf("Pod %s in namespace %s uses the image %s, expected image %s\n", pod.Name, pod.Namespace,
						container.Image, test.expectedImage)
				}
				if cpuRequest, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
					if cpuRequest.String() != test.expectedCpu {
						t.Errorf("Pod %s in namespace %s has cpu request %s, expected cpu request %s\n",
							pod.Name, pod.Namespace, cpuRequest.String(), test.expectedCpu)
					}
				}
				if memoryRequest, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
					if memoryRequest.String() != test.expectedMemery {
						t.Errorf("Pod %s in namespace %s has memory request %s, expected memory request %s\n", pod.Name,
							pod.Namespace, memoryRequest.String(), test.expectedMemery)
					}
				}
			}
		}

		// get the service object
		svcList := corev1.ServiceList{}
		err = k8sclient.List(context.Background(), &svcList, client.MatchingLabels{"app.vineyard.io/name": "vineyardd-sample"})
		if err != nil {
			t.Errorf("list services error: %v", err)
		}
		for _, svc := range svcList.Items {
			if svc.Spec.Ports[0].Port != int32(test.expectedService_port) {
				t.Errorf("Service %s in namespace %s uses the port %d, expected port %d\n", svc.Name, svc.Namespace,
					svc.Spec.Ports[0].Port, test.expectedService_port)
			}
			if string(svc.Spec.Type) != test.expectedService_type {
				t.Errorf("Service %s in namespace %s uses the type %s, expected type %s\n", svc.Name, svc.Namespace,
					string(svc.Spec.Type), test.expectedService_type)
			}
		}

	})
}

func TestValidateFormat_slow(t *testing.T) {
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

func TestGetWorkloadResource_slow(t *testing.T) {
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

func TestGetWorkloadObj_slow(t *testing.T) {
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
				fmt.Println(got)
				fmt.Println(tt.want)
				t.Errorf("GetWorkloadObj() = %v, want %v", got, tt.want)
			}
		})
	}
}

var workload = `{
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "creationTimestamp": null,
        "name": "nginx-deployment",
        "namespace": "vineyard-system",
        "ownerReferences": []
    },
    "spec": {
        "selector": {
            "matchLabels": {
                "app": "nginx"
            }
        },
        "strategy": {},
        "template": {
            "metadata": {
                "creationTimestamp": null,
                "labels": {
                    "app": "nginx",
                    "app.vineyard.io/name": "vineyard-sidecar",
                    "app.vineyard.io/role": "vineyardd"
                }
            },
            "spec": {
                "containers": [
                    {
                        "command": null,
                        "env": [
                            {
                                "name": "VINEYARD_IPC_SOCKET",
                                "value": "/var/run/vineyard.sock"
                            }
                        ],
                        "image": "nginx:1.14.2",
                        "name": "nginx",
                        "ports": [
                            {
                                "containerPort": 80
                            }
                        ],
                        "resources": {},
                        "volumeMounts": [
                            {
                                "mountPath": "/var/run",
                                "name": "vineyard-socket"
                            }
                        ]
                    },
                    {
                        "command": [
                            "/bin/bash",
                            "-c",
                            "/usr/bin/wait-for-it.sh__-t__60__vineyard-sidecar-etcd-service.vineyard-system.` +
	`svc.cluster.local:2379;__sleep__1;__/usr/local/bin/vineyardd__--sync_crds__true__` +
	`--socket__/var/run/vineyard.sock__--size____--stream_threshold__80__--etcd_cmd__etcd` +
	`__--etcd_prefix__/vineyard__--etcd_endpoint__http://vineyard-sidecar-etcd-service:2379\n"
                        ],
                        "env": [
                            {
                                "name": "VINEYARDD_UID",
                                "value": null
                            },
                            {
                                "name": "VINEYARDD_NAME",
                                "value": "vineyard-sidecar"
                            },
                            {
                                "name": "VINEYARDD_NAMESPACE",
                                "value": "vineyard-system"
                            }
                        ],
                        "image": "vineyardcloudnative/vineyardd:latest",
                        "imagePullPolicy": "IfNotPresent",
                        "name": "vineyard-sidecar",
                        "ports": [
                            {
                                "containerPort": 9600,
                                "name": "vineyard-rpc",
                                "protocol": "TCP"
                            }
                        ],
                        "resources": {
                            "limits": null,
                            "requests": null
                        },
                        "volumeMounts": [
                            {
                                "mountPath": "/var/run",
                                "name": "vineyard-socket"
                            }
                        ]
                    }
                ],
                "volumes": [
                    {
                        "emptyDir": {},
                        "name": "vineyard-socket"
                    }
                ]
            }
        }
    },
    "status": {}
}`

var rpc_service = `{
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "labels": {
            "app.vineyard.io/name": "vineyard-sidecar"
        },
        "name": "vineyard-sidecar-rpc",
        "namespace": "vineyard-system",
        "ownerReferences": [
            
        ]
    },
    "spec": {
        "ports": [
            {
                "name": "vineyard-rpc",
                "port": 9600,
                "protocol": "TCP"
            }
        ],
        "selector": {
            "app.vineyard.io/name": "vineyard-sidecar",
            "app.vineyard.io/role": "vineyardd"
        },
        "type": "ClusterIP"
    }
}`

var etcd_service = `{
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "name": "vineyard-sidecar-etcd-service",
        "namespace": "vineyard-system",
        "ownerReferences": []
    },
    "spec": {
        "ports": [
            {
                "name": "vineyard-sidecar-etcd-for-vineyard-port",
                "port": 2379,
                "protocol": "TCP",
                "targetPort": 2379
            }
        ],
        "selector": {
            "app.vineyard.io/name": "vineyard-sidecar",
            "app.vineyard.io/role": "etcd"
        }
    }
}`

var etcd_internal_service = `{
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "labels": {
            "etcd_node": "vineyard-sidecar-etcd-0"
        },
        "name": "vineyard-sidecar-etcd-0",
        "namespace": "vineyard-system",
        "ownerReferences": [
            
        ]
    },
    "spec": {
        "ports": [
            {
                "name": "client",
                "port": 2379,
                "protocol": "TCP",
                "targetPort": 2379
            },
            {
                "name": "server",
                "port": 2380,
                "protocol": "TCP",
                "targetPort": 2380
            }
        ],
        "selector": {
            "app.vineyard.io/role": "etcd",
            "etcd_node": "vineyard-sidecar-etcd-0"
        }
    }
}`

var etcd_pod = `{
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "labels": {
            "app.vineyard.io/name": "vineyard-sidecar",
            "app.vineyard.io/role": "etcd",
            "etcd_node": "vineyard-sidecar-etcd-0"
        },
        "name": "vineyard-sidecar-etcd-0",
        "namespace": "vineyard-system",
        "ownerReferences": [
            
        ]
    },
    "spec": {
        "containers": [
            {
                "command": [
                    "etcd",
                    "--name",
                    "vineyard-sidecar-etcd-0",
                    "--initial-advertise-peer-urls",
                    "http://vineyard-sidecar-etcd-0:2380",
                    "--advertise-client-urls",
                    "http://vineyard-sidecar-etcd-0:2379",
                    "--listen-peer-urls",
                    "http://0.0.0.0:2380",
                    "--listen-client-urls",
                    "http://0.0.0.0:2379",
                    "--initial-cluster",
                    "vineyard-sidecar-etcd-0=http://vineyard-sidecar-etcd-0:2380",
                    "--initial-cluster-state",
                    "new"
                ],
                "image": "vineyardcloudnative/vineyardd:latest",
                "name": "etcd",
                "ports": [
                    {
                        "containerPort": 2379,
                        "name": "client",
                        "protocol": "TCP"
                    },
                    {
                        "containerPort": 2380,
                        "name": "server",
                        "protocol": "TCP"
                    }
                ]
            }
        ],
        "restartPolicy": "Always"
    }
}`

func TestGetManifestFromTemplate_slow(t *testing.T) {
	resource := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  namespace: vineyard-system
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
`
	// Remove all line breaks and indents
	re := regexp.MustCompile(`\s+`)
	workload = strings.ReplaceAll(workload, "__", "__SPECIAL_SPACE__")
	workload = re.ReplaceAllString(workload, "")
	workload = strings.ReplaceAll(workload, "__SPECIAL_SPACE__", " ")
	rpc_service = re.ReplaceAllString(rpc_service, "")
	etcd_service = re.ReplaceAllString(etcd_service, "")
	etcd_internal_service = re.ReplaceAllString(etcd_internal_service, "")
	etcd_pod = re.ReplaceAllString(etcd_pod, "")
	type args struct {
		workload string
	}
	tests := []struct {
		name    string
		args    args
		want    OutputManifests
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Test case",
			args: args{workload: resource},
			want: OutputManifests{
				Workload:    workload + "\n",
				RPCService:  rpc_service + "\n",
				EtcdService: etcd_service + "\n",
				EtcdInternalService: []string{
					etcd_internal_service + "\n",
				},
				EtcdPod: []string{
					etcd_pod + "\n",
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetManifestFromTemplate(tt.args.workload)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetManifestFromTemplate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				fmt.Println(got.EtcdPod[0])
				//fmt.Println(etcd_service)
				t.Errorf("GetManifestFromTemplate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_deployDuringInjection(t *testing.T) {
	type args struct {
		om *OutputManifests
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			args: args{
				om: &OutputManifests{
					Workload:    workload,
					RPCService:  rpc_service,
					EtcdService: etcd_service,
					EtcdInternalService: []string{
						etcd_internal_service,
					},
					EtcdPod: []string{
						etcd_pod,
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.KubeConfig = os.Getenv("KUBECONFIG")
			if err := deployDuringInjection(tt.args.om); (err != nil) != tt.wantErr {
				t.Errorf("deployDuringInjection() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_outputInjectedResult_slow(t *testing.T) {
	type args struct {
		om OutputManifests
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			args: args{
				om: OutputManifests{
					EtcdPod: []string{
						"etcd-pod-manifest-1",
						"etcd-pod-manifest-2",
					},
					EtcdInternalService: []string{
						"etcd-internal-service-manifest-1",
					},
					EtcdService: "etcd-service-manifest",
					RPCService:  "rpc-service-manifest",
					Workload:    "workload-manifest",
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := outputInjectedResult(tt.args.om); (err != nil) != tt.wantErr {
				t.Errorf("outputInjectedResult() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_buildSidecar_slow(t *testing.T) {
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
					//ZZZ_DeprecatedClusterName:  "",
					ManagedFields: []metav1.ManagedFieldsEntry{},
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
						Size:            "256Mi",
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
						Type: "clusterIP",
						Port: 9600,
					},
				},
				Status: v1alpha1.SidecarStatus{
					Current: 0,
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := buildSidecar(tt.args.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("buildSidecar() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			a, _ := got.CreationTimestamp.Marshal()
			b, _ := tt.want.CreationTimestamp.Marshal()
			if !reflect.DeepEqual(a, b) {
				t.Errorf("buildSidecar() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInjectSidecarConfig_slow(t *testing.T) {
	sidecar := &v1alpha1.Sidecar{
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
				Size:            "256Mi",
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
				Type: "clusterIP",
				Port: 9600,
			},
		},
		Status: v1alpha1.SidecarStatus{
			Current: 0,
		},
	}

	var etcdConfig k8s.EtcdConfig
	tmplFunc := map[string]interface{}{
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
	}
	obj, _ := util.RenderManifestAsObj(string("sidecar/injection-template.yaml"), sidecar, tmplFunc)

	type args struct {
		sidecar     *v1alpha1.Sidecar
		workloadObj *unstructured.Unstructured
		sidecarObj  *unstructured.Unstructured
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			args: args{
				sidecar: sidecar,
				workloadObj: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"kind":       "Deployment",
						"apiVersion": "apps/v1",
						"metadata": map[string]interface{}{
							"name":              "nginx-deployment",
							"namespace":         "vineyard-system",
							"creationTimestamp": nil,
						},
						"spec": map[string]interface{}{
							"selector": map[string]interface{}{
								"matchLabels": map[string]interface{}{
									"app": "nginx",
								},
							},
							"template": map[string]interface{}{
								"metadata": map[string]interface{}{
									"labels": map[string]interface{}{
										"app": "nginx",
									},
									"creationTimestamp": nil,
								},
								"spec": map[string]interface{}{
									"containers": []interface{}{
										map[string]interface{}{
											"name": "nginx",
											"ports": []interface{}{
												map[string]interface{}{
													"containerPort": int64(80),
												},
											},
											"resources": map[string]interface{}{},
											"image":     "nginx:1.14.2",
										},
									},
								},
							},
							"strategy": map[string]interface{}{},
						},
						"status": map[string]interface{}{},
					},
				},
				sidecarObj: obj,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := InjectSidecarConfig(tt.args.sidecar, tt.args.workloadObj, tt.args.sidecarObj); (err != nil) != tt.wantErr {
				t.Errorf("InjectSidecarConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
