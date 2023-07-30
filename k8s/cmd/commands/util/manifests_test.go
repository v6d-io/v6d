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
	"path/filepath"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	//"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestParseManifestToObject(t *testing.T) {
	manifest := `
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  namespace: default
spec:
  containers:
  - name: my-container
    image: nginx
`
	expected := &unstructured.Unstructured{
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
	}

	result, err := ParseManifestToObject(manifest)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !reflect.DeepEqual(result, expected) {
		fmt.Println(expected)
		fmt.Println(result)
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}

func TestParseManifestsToObjects(t *testing.T) {
	manifests := []byte(`
apiVersion: v1
kind: Pod
metadata:
  name: my-pod-1
  namespace: default
spec:
  containers:
  - name: my-container
    image: nginx
`)

	expected := Manifests{
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
	}

	result, err := ParseManifestsToObjects(manifests)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(result) != len(expected) {
		t.Errorf("Expected %d objects, but got %d", len(expected), len(result))
	}

	for i := range result {
		if !reflect.DeepEqual(result[i], expected[i]) {
			fmt.Println(i)
			fmt.Println(expected[i])
			fmt.Println(result[i])
			t.Errorf("Expected %v, but got %v", expected[i], result[i])
		}
	}
}

func TestApplyManifests(t *testing.T) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	kubeconfig := filepath.Join(homeDir, ".kube", "config")
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)

	clientScheme := runtime.NewScheme()
	//_ = scheme.AddToScheme(clientScheme)
	client, err := client.New(config, client.Options{Scheme: clientScheme})

	// 创建要应用的 Manifests
	manifests := Manifests{
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
	}

	// 指定要使用的命名空间
	namespace := "vineyard-system"

	// 调用 ApplyManifests
	err = ApplyManifests(client, manifests, namespace)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

}

func TestDeleteManifests(t *testing.T) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	kubeconfig := filepath.Join(homeDir, ".kube", "config")
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)

	clientScheme := runtime.NewScheme()
	//_ = scheme.AddToScheme(clientScheme)
	client, err := client.New(config, client.Options{Scheme: clientScheme})

	// 创建要应用的 Manifests
	manifests := Manifests{
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
	}

	// 指定要使用的命名空间
	namespace := "vineyard-system"

	// 调用 DeleteManifests
	err = DeleteManifests(client, manifests, namespace)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
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
