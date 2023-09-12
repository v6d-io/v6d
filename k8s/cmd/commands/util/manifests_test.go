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
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

func Test_ParseManifestToObject(t *testing.T) {
	type args struct {
		manifest string
	}
	tests := []struct {
		name    string
		args    args
		want    *unstructured.Unstructured
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				manifest: `
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  namespace: default
spec:
  containers:
  - name: my-container
    image: nginx
`,
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
					"metadata": map[string]interface{}{
						"name":              "my-pod",
						"namespace":         "default",
						"creationTimestamp": nil,
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":      "my-container",
								"image":     "nginx",
								"resources": map[string]interface{}{},
							},
						},
					},
					"status": map[string]interface{}{},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseManifestToObject(tt.args.manifest)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseManifestToObject() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseManifestToObject() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_ParseManifestsToObjects(t *testing.T) {
	type args struct {
		manifests []byte
	}
	tests := []struct {
		name    string
		args    args
		want    Manifests
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				manifests: []byte(`
apiVersion: v1
kind: Pod
metadata:
  name: my-pod-1
  namespace: default
spec:
  containers:
  - name: my-container
    image: nginx
`),
			},
			want: Manifests{
				&unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "Pod",
						"metadata": map[string]interface{}{
							"creationTimestamp": nil,
							"name":              "my-pod-1",
							"namespace":         "default",
						},
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"name":      "my-container",
									"image":     "nginx",
									"resources": map[string]interface{}{},
								},
							},
						},
						"status": map[string]interface{}{},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseManifestsToObjects(tt.args.manifests)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseManifestsToObjects() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				for i := range tt.want {
					if !reflect.DeepEqual(got[i], tt.want[i]) {
						t.Errorf("Expected %v, but got %v", tt.want[i], got[i])
					}
				}
			}
		})
	}
}

func Test_ApplyManifests(t *testing.T) {
	flags.KubeConfig = kube_config
	c := KubernetesClient()

	type args struct {
		c         client.Client
		manifests Manifests
		namespace string
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "Test case 1",
			args: args{
				c: c,
				manifests: Manifests{
					&unstructured.Unstructured{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"creationTimestamp": nil,
								"name":              "my-pod-1",
								"namespace":         "vineyard-system",
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":      "my-container",
										"image":     "hello-world",
										"resources": map[string]interface{}{},
									},
								},
							},
							"status": map[string]interface{}{},
						},
					},
				},
				namespace: "vineyard-system",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ApplyManifests(tt.args.c, tt.args.manifests, tt.args.namespace); err != nil {
				t.Errorf("ApplyManifests() error = %v", err)
			}
		})
		pod := &unstructured.Unstructured{}
		pod.SetKind("Pod")
		pod.SetAPIVersion("v1")
		err := c.Get(context.Background(), client.ObjectKey{Name: "my-pod-1", Namespace: "vineyard-system"}, pod)
		assert.Nil(t, err)

	}
}

func Test_DeleteManifests(t *testing.T) {
	flags.KubeConfig = kube_config
	c := KubernetesClient()

	type args struct {
		c         client.Client
		manifests Manifests
		namespace string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				c: c,
				manifests: Manifests{
					&unstructured.Unstructured{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"creationTimestamp": nil,
								"name":              "my-pod-1",
								"namespace":         "vineyard-system",
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":      "my-container",
										"image":     "hello-world",
										"resources": map[string]interface{}{},
									},
								},
							},
							"status": map[string]interface{}{},
						},
					},
				},
				namespace: "vineyard-system",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := DeleteManifests(tt.args.c, tt.args.manifests, tt.args.namespace); (err != nil) != tt.wantErr {
				t.Errorf("DeleteManifests() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
		pod := &unstructured.Unstructured{}
		pod.SetKind("Pod")
		pod.SetAPIVersion("v1")
		err := c.Get(context.Background(), client.ObjectKey{Name: "my-pod-1", Namespace: "vineyard-system"}, pod)
		assert.NotNil(t, err)
	}
}

func Test_ApplyManifestsWithOwnerRef(t *testing.T) {
	flags.KubeConfig = kube_config
	flags.Namespace = "vineyard-system"
	c := KubernetesClient()

	objs := []*unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "batch/v1",
				"kind":       "Job",
				"metadata": map[string]interface{}{
					"name":      "vineyard-backup",
					"namespace": "vineyard-system",
				},
				"spec": map[string]interface{}{
					"parallelism": int64(3),
					"template": map[string]interface{}{
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app.kubernetes.io/name": "vineyard-backup",
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
															"vineyard-backup",
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
											"name":  "BACKUP_PATH",
											"value": "/var/vineyard/dump",
										},
										map[string]interface{}{
											"name":  "ENDPOINT",
											"value": "vineyardd-sample-rpc.vineyard-system",
										},
										map[string]interface{}{
											"name":  "SELECTOR",
											"value": "vineyard-backup",
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
									"image":           "ghcr.io/v6d-io/v6d/backup-job",
									"imagePullPolicy": "IfNotPresent",
									"name":            "engine",
									"volumeMounts": []interface{}{
										map[string]interface{}{
											"mountPath": "/var/run",
											"name":      "vineyard-sock",
										},
										map[string]interface{}{
											"mountPath": "/var/vineyard/dump",
											"name":      "backup-path",
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
									"name": "backup-path",
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
						"app.kubernetes.io/name": "backup",
					},
					"name":      "vineyard-backup",
					"namespace": "vineyard-system",
				},
				"roleRef": map[string]interface{}{
					"apiGroup": "rbac.authorization.k8s.io",
					"kind":     "Role",
					"name":     "vineyard-backup",
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
						"app.kubernetes.io/instance": "backup",
					},
					"name":      "vineyard-backup",
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
	}

	// Test with valid inputs
	err := ApplyManifestsWithOwnerRef(c, objs, "Job", "Role,Rolebinding")
	assert.Nil(t, err)

	// Check if OwnerReference was set for "Role" kind
	RoleBinding := &unstructured.Unstructured{}
	RoleBinding.SetKind("Role")
	RoleBinding.SetAPIVersion("rbac.authorization.k8s.io/v1")

	err = c.Get(context.Background(), client.ObjectKey{Name: "vineyard-backup", Namespace: "vineyard-system"}, RoleBinding)
	assert.Nil(t, err)
	time.Sleep(2 * time.Second)
	ownerRefs := RoleBinding.GetOwnerReferences()
	assert.Equal(t, 1, len(ownerRefs))
	assert.Equal(t, "Job", ownerRefs[0].Kind)
	assert.Equal(t, "vineyard-backup", ownerRefs[0].Name)

	// Test with unsupported kind in refKind
	err = ApplyManifestsWithOwnerRef(c, objs, "Job", "Unsupported")
	assert.Nil(t, err)
}
