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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"fmt"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

/*func TestNewDeployRecoverJobCmd(t *testing.T) {
	tests := []struct {
		name string
		want *cobra.Command
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: deployRecoverJobCmd, // 指定预期的 *cobra.Command 值
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewDeployRecoverJobCmd(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewDeployRecoverJobCmd() = %v, want %v", got, tt.want)
			}
		})
	}
}*/

func Test_getRecoverObjectsFromTemplate(t *testing.T) {
	flags.KubeConfig = "/home/zhuyi/.kube/config"
	c := util.KubernetesClient()

	type args struct {
		c client.Client
	}
	tests := []struct {
		name    string
		args    args
		want    []*unstructured.Unstructured
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Test case 1",
			args: args{
				c: c,
			},
			want: []*unstructured.Unstructured{
				{
					Object: map[string]interface{}{
						"apiVersion": "batch/v1",
						"kind":       "Job",
						"metadata": map[string]interface{}{
							"name":      "vineyard-recover",
							"namespace": "vineyard-system",
						},
						"spec": map[string]interface{}{
							"parallelism": int64(1),
							"template": map[string]interface{}{
								"metadata": map[string]interface{}{
									"labels": map[string]interface{}{
										"app.kubernetes.io/name": "vineyard-recover",
									},
								},
								"spec": map[string]interface{}{
									"affinity": map[string]interface{}{
										"podAffinity": map[string]interface{}{
											"requiredDuringSchedulingIgnoredDuringExecution": []interface{}{
												map[string]interface{}{
													"labelSelector": map[string]interface{}{
														"matchExpressions": []interface{}{
															map[string]interface{}{
																"key":      "app.kubernetes.io/instance",
																"operator": "In",
																"values": []interface{}{
																	"vineyard-system-vineyard-operator-cert-manager",
																},
															},
														},
													},
													"topologyKey": "kubernetes.io/hostname",
												},
											},
										},
										"podAntiAffinity": map[string]interface{}{
											"requiredDuringSchedulingIgnoredDuringExecution": []interface{}{
												map[string]interface{}{
													"labelSelector": map[string]interface{}{
														"matchExpressions": []interface{}{
															map[string]interface{}{
																"key":      "app.kubernetes.io/name",
																"operator": "In",
																"values": []interface{}{
																	"vineyard-recover",
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
													"name":  "RECOVER_PATH",
													"value": nil,
												},
												map[string]interface{}{
													"name":  "ENDPOINT",
													"value": "vineyard-operator-cert-manager-rpc.vineyard-system",
												},
												map[string]interface{}{
													"name":  "SELECTOR",
													"value": "vineyard-recover",
												},
												map[string]interface{}{
													"name":  "ALLINSTANCES",
													"value": "1",
												},
												map[string]interface{}{
													"name": "POD_NAME",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "metadata.name",
														},
													},
												},
												map[string]interface{}{
													"name": "POD_NAMESPACE",
													"valueFrom": map[string]interface{}{
														"fieldRef": map[string]interface{}{
															"fieldPath": "metadata.namespace",
														},
													},
												},
											},
											"image":           "ghcr.io/v6d-io/v6d/recover-job",
											"imagePullPolicy": "IfNotPresent",
											"name":            "engine",
											"volumeMounts": []interface{}{
												map[string]interface{}{
													"mountPath": "/var/run",
													"name":      "vineyard-sock",
												},
												map[string]interface{}{
													"mountPath": nil,
													"name":      "recover-path",
												},
											},
										},
									},
									"restartPolicy": "Never",
									"volumes": []interface{}{
										map[string]interface{}{
											"hostPath": map[string]interface{}{
												"path": "/var/run/vineyard-kubernetes/vineyard-system/vineyard-operator-cert-manager",
											},
											"name": "vineyard-sock",
										},
										map[string]interface{}{
											"name": "recover-path",
											"persistentVolumeClaim": map[string]interface{}{
												"claimName": "vineyard-backup",
											},
										},
									},
								},
							},
							"ttlSecondsAfterFinished": int64(80),
						},
					},
				},
				{
					Object: map[string]interface{}{
						"apiVersion": "rbac.authorization.k8s.io/v1",
						"kind":       "RoleBinding",
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app.kubernetes.io/name": "recover",
							},
							"name":      "vineyard-recover",
							"namespace": "vineyard-system",
						},
						"roleRef": map[string]interface{}{
							"apiGroup": "rbac.authorization.k8s.io",
							"kind":     "Role",
							"name":     "vineyard-recover",
						},
						"subjects": []interface{}{
							map[string]interface{}{
								"kind":      "ServiceAccount",
								"name":      "default",
								"namespace": "vineyard-system",
							},
						},
					},
				},
				{
					Object: map[string]interface{}{
						"apiVersion": "rbac.authorization.k8s.io/v1",
						"kind":       "Role",
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app.kubernetes.io/instance": "recover",
							},
							"name":      "vineyard-recover",
							"namespace": "vineyard-system",
						},
						"rules": []interface{}{
							map[string]interface{}{
								"apiGroups": []interface{}{
									"",
								},
								"resources": []interface{}{
									"pods",
									"pods/log",
								},
								"verbs": []interface{}{
									"get",
									"list",
								},
							},
							map[string]interface{}{
								"apiGroups": []interface{}{
									"",
								},
								"resources": []interface{}{
									"pods/exec",
								},
								"verbs": []interface{}{
									"create",
								},
							},
						},
						"subjects": []interface{}{
							map[string]interface{}{
								"kind":      "ServiceAccount",
								"name":      "default",
								"namespace": "vineyard-system",
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.VineyardDeploymentName = "vineyard-operator-cert-manager"
			flags.VineyardDeploymentNamespace = "vineyard-system"
			flags.Namespace = "vineyard-system"
			got, err := getRecoverObjectsFromTemplate(tt.args.c)
			if (err != nil) != tt.wantErr {
				t.Errorf("getRecoverObjectsFromTemplate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			for i := range got {
				if !reflect.DeepEqual(*got[i], *(tt.want)[i]) {
					fmt.Println(i)
					fmt.Println(*got[i])
					fmt.Println(*(tt.want)[i])
					t.Errorf("getRecoverObjectsFromTemplate() = %+v, want %+v", got, tt.want)

				}
			}
		})
	}
}

/*func Test_waitRecoverJobReady(t *testing.T) {
	flags.KubeConfig = "/home/zhuyi/.kube/config"
	c := util.KubernetesClient()
	type args struct {
		c client.Client
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Job succeeded",
			args: args{
				c: c,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.Namespace = "vineyard-system"
			if err := waitRecoverJobReady(tt.args.c); (err != nil) != tt.wantErr {
				t.Errorf("waitRecoverJobReady() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// not implemented
func Test_createMappingTableConfigmap(t *testing.T) {
	flags.KubeConfig = "/home/zhuyi/.kube/config"
	c := util.KubernetesClient()

	config, _ := clientcmd.BuildConfigFromFlags("", flags.KubeConfig)

	cs, _ := kubernetes.NewForConfig(config)

	type args struct {
		c  client.Client
		cs kubernetes.Clientset
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
				c:  c,
				cs: *cs,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.Namespace = "vineyard-system"
			if err := createMappingTableConfigmap(tt.args.c, tt.args.cs); (err != nil) != tt.wantErr {
				t.Errorf("createMappingTableConfigmap() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}*/
