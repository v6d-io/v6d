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

package deploy

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/spf13/cobra"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

func validatePodAttributes(t *testing.T, pods corev1.PodList, expectedImage, expectedCpu, expectedMemory string, vineyardReplicas int) {
	for _, pod := range pods.Items {
		for _, container := range pod.Spec.Containers {
			if container.Image != expectedImage {
				t.Errorf("Pod %s in namespace %s uses the image %s, expected image %s\n",
					pod.Name, pod.Namespace, container.Image, expectedImage)
			}
			if cpuRequest, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
				if cpuRequest.String() != expectedCpu {
					t.Errorf("Pod %s in namespace %s has cpu request %s, expected cpu request %s\n",
						pod.Name, pod.Namespace, cpuRequest.String(), expectedCpu)
				}
			}
			if memoryRequest, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
				if memoryRequest.String() != expectedMemory {
					t.Errorf("Pod %s in namespace %s has memory request %s, expected memory request %s\n",
						pod.Name, pod.Namespace, memoryRequest.String(), expectedMemory)
				}
			}

		}
	}
	if len(pods.Items) != vineyardReplicas {
		t.Errorf("vineyardd replicas want: %d, got: %d", vineyardReplicas, len(pods.Items))
	}
}

func TestDeployVineyardDeploymentCmd_DeployVineyarddCmd_first(t *testing.T) {
	test := []struct {
		name                 string
		vineyardReplicas     int
		etcdReplicas         int
		expectedImage        string
		expectedCpu          string
		expectedMemory       string
		expectedService_port int
		expectedService_type string
		test_cmd             *cobra.Command
	}{
		{
			name:                 "test replicas",
			vineyardReplicas:     3,
			etcdReplicas:         1,
			expectedImage:        "vineyardcloudnative/vineyardd:latest",
			expectedCpu:          "",
			expectedMemory:       "",
			expectedService_port: 9600,
			expectedService_type: "ClusterIP",
			test_cmd:             deployVineyarddCmd,
		},
		{
			name:                 "test replicas",
			vineyardReplicas:     3,
			etcdReplicas:         1,
			expectedImage:        "vineyardcloudnative/vineyardd:latest",
			expectedCpu:          "",
			expectedMemory:       "",
			expectedService_port: 9600,
			expectedService_type: "ClusterIP",
			test_cmd:             deployVineyardDeploymentCmd,
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			// set the flags
			flags.Namespace = vineyard_default_namespace
			flags.KubeConfig = kube_config
			flags.VineyarddOpts.Replicas = 3
			flags.VineyarddOpts.EtcdReplicas = 1
			flags.VineyarddOpts.Vineyard.Image = vineyard_image
			flags.VineyarddOpts.Vineyard.Memory = ""
			flags.VineyarddOpts.Service.Port = 9600
			flags.VineyarddOpts.Service.Type = service_type
			flags.VineyarddOpts.Vineyard.Size = size
			tt.test_cmd.Run(tt.test_cmd, []string{})
			time.Sleep(1 * time.Second)
			// get the replicas of etcd and vineyardd
			k8sclient := util.KubernetesClient()
			vineyardPods := corev1.PodList{}
			etcdPod := corev1.PodList{}
			vineyarddOpts := []client.ListOption{
				client.InNamespace(flags.Namespace),
				client.MatchingLabels{
					"app.vineyard.io/name": flags.VineyarddName,
					"app.vineyard.io/role": "vineyardd",
				},
			}
			err := k8sclient.List(context.Background(), &vineyardPods, vineyarddOpts...)
			if err != nil {
				t.Errorf("list vineyardd pods error: %v", err)
			}
			etcdOpts := []client.ListOption{
				client.InNamespace(flags.Namespace),
				client.MatchingLabels{
					"app.vineyard.io/name": flags.VineyarddName,
					"app.vineyard.io/role": "etcd",
				},
			}
			err = k8sclient.List(context.Background(), &etcdPod, etcdOpts...)
			if err != nil {
				t.Errorf("list etcd pods error: %v", err)
			}

			validatePodAttributes(t, vineyardPods, tt.expectedImage, tt.expectedCpu, tt.expectedMemory, tt.vineyardReplicas)
			validatePodAttributes(t, etcdPod, tt.expectedImage, tt.expectedCpu, tt.expectedMemory, tt.etcdReplicas)

			// get the service object
			svcList := corev1.ServiceList{}
			err = k8sclient.List(context.Background(), &svcList, client.MatchingLabels{"app.vineyard.io/name": "vineyardd-sample"})
			if err != nil {
				t.Errorf("list services error: %v", err)
			}
			for _, svc := range svcList.Items {
				if svc.Spec.Ports[0].Port != int32(tt.expectedService_port) {
					t.Errorf("Service %s in namespace %s uses the port %d, expected port %d\n",
						svc.Name, svc.Namespace, svc.Spec.Ports[0].Port, tt.expectedService_port)
				}
				if string(svc.Spec.Type) != tt.expectedService_type {
					t.Errorf("Service %s in namespace %s uses the type %s, expected type %s\n",
						svc.Name, svc.Namespace, string(svc.Spec.Type), tt.expectedService_type)
				}
			}
		})
	}

}

func TestGetVineyardDeploymentObjectsFromTemplate_third(t *testing.T) {
	// set different flags
	flags.Namespace = "test-vineyard-system"
	flags.VineyarddOpts.Replicas = 10
	flags.VineyarddName = "test-vineyardd-sample"

	tests := []struct {
		name    string
		want    []*unstructured.Unstructured
		wantErr bool
	}{
		// Add test cases.
		{
			name: "test case",
			want: []*unstructured.Unstructured{
				{
					Object: map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "Pod",
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app.vineyard.io/name": "test-vineyardd-sample",
								"app.vineyard.io/role": "etcd",
								"etcd_node":            "test-vineyardd-sample-etcd-0",
							},
							"name":      "test-vineyardd-sample-etcd-0",
							"namespace": "test-vineyard-system",
						},
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"command": []interface{}{
										"etcd",
										"--name",
										"test-vineyardd-sample-etcd-0",
										"--initial-advertise-peer-urls",
										"http://test-vineyardd-sample-etcd-0:2380",
										"--advertise-client-urls",
										"http://test-vineyardd-sample-etcd-0:2379",
										"--listen-peer-urls",
										"http://0.0.0.0:2380",
										"--listen-client-urls",
										"http://0.0.0.0:2379",
										"--initial-cluster",
										"test-vineyardd-sample-etcd-0=http://test-vineyardd-sample-etcd-0:2380",
										"--initial-cluster-state",
										"new",
									},
									"image":           "vineyardcloudnative/vineyardd:latest",
									"imagePullPolicy": "IfNotPresent",
									"name":            "etcd",
									"ports": []interface{}{
										map[string]interface{}{
											"containerPort": int64(2379),
											"name":          "client",
											"protocol":      "TCP",
										},
										map[string]interface{}{
											"containerPort": int64(2380),
											"name":          "server",
											"protocol":      "TCP",
										},
									},
								},
							},

							"restartPolicy": "Always",
						},
					},
				},
				{
					Object: map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "Service",
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"etcd_node": "test-vineyardd-sample-etcd-0",
							},
							"name":      "test-vineyardd-sample-etcd-0",
							"namespace": "test-vineyard-system",
						},
						"spec": map[string]interface{}{
							"ports": []interface{}{
								map[string]interface{}{
									"name":       "client",
									"port":       int64(2379),
									"protocol":   "TCP",
									"targetPort": int64(2379),
								},
								map[string]interface{}{
									"name":       "server",
									"port":       int64(2380),
									"protocol":   "TCP",
									"targetPort": int64(2380),
								},
							},
							"selector": map[string]interface{}{
								"app.vineyard.io/role": "etcd",
								"etcd_node":            "test-vineyardd-sample-etcd-0",
							},
						},
					},
				},
				{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app.kubernetes.io/component": "deployment",
								"app.kubernetes.io/instance":  "test-vineyard-system-test-vineyardd-sample",
								"app.vineyard.io/name":        "test-vineyardd-sample",
							},
							"name":      "test-vineyardd-sample",
							"namespace": "test-vineyard-system",
						},
						"spec": map[string]interface{}{
							"replicas": int64(10),
							"selector": map[string]interface{}{
								"matchLabels": map[string]interface{}{
									"app.kubernetes.io/instance": "test-vineyard-system-test-vineyardd-sample",
									"app.kubernetes.io/name":     "test-vineyardd-sample",
									"app.vineyard.io/name":       "test-vineyardd-sample",
								},
							},
							"template": map[string]interface{}{
								"metadata": map[string]interface{}{
									"annotations": map[string]interface{}{
										"kubectl.kubernetes.io/default-container":      "vineyardd",
										"kubectl.kubernetes.io/default-logs-container": "vineyardd",
									},
									"labels": map[string]interface{}{
										"app.kubernetes.io/component": "deployment",
										"app.kubernetes.io/instance":  "test-vineyard-system-test-vineyardd-sample",
										"app.kubernetes.io/name":      "test-vineyardd-sample",
										"app.vineyard.io/name":        "test-vineyardd-sample",
										"app.vineyard.io/role":        "vineyardd",
									},
								},
								"spec": map[string]interface{}{
									"affinity": map[string]interface{}{
										"podAntiAffinity": map[string]interface{}{
											"requiredDuringSchedulingIgnoredDuringExecution": []interface{}{
												map[string]interface{}{
													"labelSelector": map[string]interface{}{
														"matchExpressions": []interface{}{
															map[string]interface{}{
																"key":      "app.kubernetes.io/instance",
																"operator": "In",
																"values": []interface{}{
																	"test-vineyard-system-test-vineyardd-sample",
																},
															},
														},
													},
													"topologyKey": "kubernetes.io/hostname",
												},
											},
										},
									},
									"containers": []interface{}{
										map[string]interface{}{
											"env": []interface{}{
												map[string]interface{}{
													"name":  "VINEYARDD_UID",
													"value": nil,
												},
												map[string]interface{}{
													"name":  "VINEYARDD_NAME",
													"value": "test-vineyardd-sample",
												},
												map[string]interface{}{
													"name":  "VINEYARDD_NAMESPACE",
													"value": "test-vineyard-system",
												},
												map[string]interface{}{
													"name": "MY_NODE_NAME",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "spec.nodeName",
														},
													},
												},
												map[string]interface{}{
													"name": "MY_POD_NAME",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "metadata.name",
														},
													},
												},
												map[string]interface{}{
													"name": "MY_POD_NAMESPACE",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "metadata.namespace",
														},
													},
												},
												map[string]interface{}{
													"name": "MY_UID",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "metadata.uid",
														},
													},
												},
												map[string]interface{}{
													"name": "MY_POD_IP",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "status.podIP",
														},
													},
												},
												map[string]interface{}{
													"name": "MY_HOST_NAME",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "status.podIP",
														},
													},
												},
												map[string]interface{}{
													"name": "USER",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "metadata.name",
														},
													},
												},
											},
											"readinessProbe": map[string]interface{}{
												"exec": map[string]interface{}{
													"command": []interface{}{
														"ls",
														"/var/run/vineyard.sock",
													},
												},
											},
											"resources": map[string]interface{}{
												"limits":   nil,
												"requests": nil,
											},
											"securityContext": map[string]interface{}{},
											"command": []interface{}{
												"/bin/bash",
												"-c",
												"/usr/local/bin/vineyardd --sync_crds true --socket " +
													"/var/run/vineyard.sock --size  --stream_threshold 80 --etcd_cmd etcd --etcd_prefix " +
													"/vineyard --etcd_endpoint http://test-vineyardd-sample-etcd-service:2379\n",
											},
											"image":           "vineyardcloudnative/vineyardd:latest",
											"imagePullPolicy": "IfNotPresent",
											"livenessProbe": map[string]interface{}{
												"periodSeconds": int64(60),
												"tcpSocket": map[string]interface{}{
													"port": int64(9600),
												},
											},
											"name": "vineyardd",
											"ports": []interface{}{
												map[string]interface{}{
													"containerPort": int64(9600),
													"name":          "rpc",
													"protocol":      "TCP",
												},
											},
											"volumeMounts": []interface{}{
												map[string]interface{}{
													"mountPath": "/var/run",
													"name":      "vineyard-socket",
												},
												map[string]interface{}{
													"mountPath": "/dev/shm",
													"name":      "shm",
												},
												map[string]interface{}{
													"mountPath": "/var/log/vineyard",
													"name":      "log",
												},
											},
										},
									},
									"volumes": []interface{}{
										map[string]interface{}{
											"hostPath": map[string]interface{}{
												"path": "/var/run/vineyard-kubernetes/test-vineyard-system/test-vineyardd-sample",
											},
											"name": "vineyard-socket",
										},
										map[string]interface{}{
											"emptyDir": map[string]interface{}{
												"medium": "Memory",
											},
											"name": "shm",
										},
										map[string]interface{}{
											"emptyDir": map[string]interface{}{},
											"name":     "log",
										},
									},
								},
							},
						},
					},
				},
				{
					Object: map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "Service",
						"metadata": map[string]interface{}{
							"name":      "test-vineyardd-sample-etcd-service",
							"namespace": "test-vineyard-system",
						},
						"spec": map[string]interface{}{
							"ports": []interface{}{
								map[string]interface{}{
									"name":       "etcd-for-vineyard-port",
									"port":       int64(2379),
									"protocol":   "TCP",
									"targetPort": int64(2379),
								},
							},
							"selector": map[string]interface{}{
								"app.vineyard.io/name": "test-vineyardd-sample",
								"app.vineyard.io/role": "etcd",
							},
						},
					},
				},
				{
					Object: map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "Service",
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app.vineyard.io/name": "test-vineyardd-sample",
							},
							"name":      "test-vineyardd-sample-rpc",
							"namespace": "test-vineyard-system",
						},
						"spec": map[string]interface{}{
							"ports": []interface{}{
								map[string]interface{}{
									"name":     "vineyard-rpc",
									"port":     int64(9600),
									"protocol": "TCP",
								},
							},
							"selector": map[string]interface{}{
								"app.vineyard.io/name": "test-vineyardd-sample",
								"app.vineyard.io/role": "vineyardd",
							},
							"type": "ClusterIP",
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetVineyardDeploymentObjectsFromTemplate()
			if (err != nil) != tt.wantErr {
				t.Errorf("GetVineyardDeploymentObjectsFromTemplate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			for i := range got {
				gotStr := fmt.Sprintf("%v", got[i])
				wantStr := fmt.Sprintf("%v", tt.want[i])
				if !reflect.DeepEqual(gotStr, wantStr) {
					t.Errorf("getDeploymentObjectsFromTemplate() = %+v, want %+v", gotStr, wantStr)

				}
			}
		})
	}
}
