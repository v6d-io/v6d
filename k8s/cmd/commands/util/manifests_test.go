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
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	//"k8s.io/client-go/kubernetes/scheme"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
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
				fmt.Println(tt.want)
				fmt.Println(got)
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
						fmt.Println(i)
						fmt.Println(tt.want[i])
						fmt.Println(got[i])
						t.Errorf("Expected %v, but got %v", tt.want[i], got[i])
					}
				}
			}
		})
	}
}

func Test_ApplyManifests(t *testing.T) {
	flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
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
										"image":     "nginx",
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
	}
}

func Test_DeleteManifests(t *testing.T) {
	flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
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
										"image":     "nginx",
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
	}
}

func TestApplyManifestsWithOwnerRef(t *testing.T) {
	scheme := runtime.NewScheme()

	c := fake.NewFakeClientWithScheme(scheme)

	objs := []*unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "batch/v1",
				"kind":       "Job",
				"metadata": map[string]interface{}{
					"name": "test-job",
					"uid":  "12345",
				},
			},
		},
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name": "test-pod",
				},
			},
		},
	}

	// Test with valid inputs
	err := ApplyManifestsWithOwnerRef(c, objs, "Job", "Pod")
	assert.Nil(t, err)

	// Check if OwnerReference was set for "Pod" kind
	pod := &unstructured.Unstructured{}
	pod.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "Pod",
	})

	err = c.Get(context.Background(), client.ObjectKey{Name: "test-pod"}, pod)
	assert.Nil(t, err)

	ownerRefs := pod.GetOwnerReferences()
	assert.Equal(t, 1, len(ownerRefs))
	assert.Equal(t, "Job", ownerRefs[0].Kind)
	assert.Equal(t, "test-job", ownerRefs[0].Name)
	assert.Equal(t, "12345", string(ownerRefs[0].UID))

	// Test with unsupported kind in refKind
	err = ApplyManifestsWithOwnerRef(c, objs, "Job", "Unsupported")
	assert.Nil(t, err)
}

/*func Test_ApplyManifestsWithOwnerRef(t *testing.T) {
	flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
	//flags.Namespace = "vineyard-system"
	c := KubernetesClient()

	objs := []*unstructured.Unstructured{
		{
			Object: map[string]interface{}{
				"apiVersion": "batch/v1",
				"kind":       "Job",
				"metadata": map[string]interface{}{
					"name":      "test-job",
					"uid":       "12345",
					"namespace": "vineyard-system",
				},
			},
		},
		{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":      "test-pod",
					"namespace": "vineyard-system",
				},
			},
		},
	}

	// Test with valid inputs
	err := ApplyManifestsWithOwnerRef(c, objs, "Job", "Pod")
	assert.Nil(t, err)

	// Check if OwnerReference was set for "Pod" kind
	pod := &unstructured.Unstructured{}
	pod.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "Pod",
	})

	err = c.Get(context.Background(), client.ObjectKey{Name: "test-pod"}, pod)
	assert.Nil(t, err)

	ownerRefs := pod.GetOwnerReferences()
	assert.Equal(t, 1, len(ownerRefs))
	assert.Equal(t, "Job", ownerRefs[0].Kind)
	assert.Equal(t, "test-job", ownerRefs[0].Name)
	assert.Equal(t, "12345", string(ownerRefs[0].UID))

	// Test with unsupported kind in refKind
	err = ApplyManifestsWithOwnerRef(c, objs, "Job", "Unsupported")
	assert.Nil(t, err)
}*/
