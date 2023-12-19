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
	"reflect"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

func TestDeployRecoverJobCmd_second(t *testing.T) {
	// deploy a vineyardd for later backup operation
	flags.KubeConfig = kube_config
	flags.Namespace = vineyard_default_namespace
	flags.VineyarddOpts.Replicas = 3
	flags.VineyarddOpts.EtcdReplicas = 1
	flags.VineyarddOpts.Vineyard.Image = vineyard_image
	flags.VineyarddOpts.Vineyard.CPU = ""
	flags.VineyarddOpts.Vineyard.Memory = ""
	flags.VineyarddOpts.Service.Port = 9600
	flags.VineyarddOpts.Service.Type = service_type
	flags.VineyarddOpts.Volume.PvcName = ""
	flags.VineyarddOpts.Vineyard.Size = size
	deployVineyardDeploymentCmd := NewDeployVineyardDeploymentCmd()
	deployVineyardDeploymentCmd.Run(deployVineyardDeploymentCmd, []string{})

	// recover operation
	flags.RecoverPath = backup_path
	flags.PVCName = "vineyard-backup"
	flags.VineyardDeploymentName = vineyard_deployment_name
	flags.VineyardDeploymentNamespace = vineyard_deployment_namespace
	deployRecoverJobCmd.Run(deployRecoverJobCmd, []string{})
	c := util.KubernetesClient()

	if util.Wait(func() (bool, error) {
		jobName := flags.RecoverName
		name := client.ObjectKey{Name: jobName, Namespace: flags.Namespace}
		job := batchv1.Job{}
		if err := c.Get(context.TODO(), name, &job); err != nil {
			return false, err
		}
		if job.Status.Succeeded == *job.Spec.Parallelism {
			return true, nil
		}
		return false, nil
	}) != nil {
		t.Errorf("recover job can not be deployed successfully")
	}

	// get the ConfigMap
	name := client.ObjectKey{Name: flags.RecoverName + "-mapping-table", Namespace: flags.Namespace}
	cm := corev1.ConfigMap{}
	if err := c.Get(context.TODO(), name, &cm); err != nil {
		t.Errorf("can not get MappingTableConfigmap")
	}
}

func Test_getRecoverObjectsFromTemplate_third(t *testing.T) {
	// set the flags
	flags.KubeConfig = kube_config
	flags.VineyardDeploymentName = vineyard_deployment_name
	flags.VineyardDeploymentNamespace = vineyard_deployment_namespace
	flags.Namespace = vineyard_default_namespace
	flags.PVCName = "vineyard-backup"
	flags.RecoverPath = backup_path
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
		// Add test cases.
		{
			name: "Test case",
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
							"parallelism": int64(3),
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
																	"vineyard-system-vineyardd-sample",
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
													"value": "/var/vineyard/dump",
												},
												map[string]interface{}{
													"name":  "ENDPOINT",
													"value": "vineyardd-sample-rpc.vineyard-system",
												},
												map[string]interface{}{
													"name":  "SELECTOR",
													"value": "vineyard-recover",
												},
												map[string]interface{}{
													"name":  "ALLINSTANCES",
													"value": "3",
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
													"mountPath": "/var/vineyard/dump",
													"name":      "recover-path",
												},
											},
										},
									},
									"restartPolicy": "Never",
									"volumes": []interface{}{
										map[string]interface{}{
											"hostPath": map[string]interface{}{
												"path": "/var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample",
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
			got, err := getRecoverObjectsFromTemplate(tt.args.c)
			if (err != nil) != tt.wantErr {
				t.Errorf("getRecoverObjectsFromTemplate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			for i := range got {
				if !reflect.DeepEqual(*got[i], *(tt.want)[i]) {
					t.Errorf("getRecoverObjectsFromTemplate() = %+v, want %+v", got, tt.want)
				}
			}
		})
	}
}
