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
	"os"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

var kube_config = os.Getenv("KUBECONFIG")

func Test_Scheme(t *testing.T) {
	expectedScheme := scheme

	tests := []struct {
		name string
		want *runtime.Scheme
	}{
		{
			name: "Test case 1",
			want: expectedScheme,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			if got := Scheme(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Scheme() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_Create(t *testing.T) {
	type args struct {
		c     client.Client
		v     *corev1.ConfigMap
		until []func(*corev1.ConfigMap) bool
	}

	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				c: fake.NewClientBuilder().Build(),
				v: &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "my-configmap",
						Namespace: "default",
					},
					Data: map[string]string{
						"key": "value",
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Create(tt.args.c, tt.args.v, tt.args.until...); (err != nil) != tt.wantErr {
				t.Errorf("Create() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Validate the created object
			createdConfigMap := &corev1.ConfigMap{}
			err := tt.args.c.Get(context.TODO(), types.NamespacedName{
				Name:      "my-configmap",
				Namespace: "default",
			}, createdConfigMap)
			if err != nil {
				t.Errorf("Failed to retrieve the created object: %v", err)
			}

			// Verify the object's name and namespace match the expected values
			if createdConfigMap.Name != tt.args.v.GetName() {
				t.Errorf("Object names should match. Expected: %s, Got: %s", tt.args.v.GetName(), createdConfigMap.Name)
			}
			if createdConfigMap.Namespace != tt.args.v.GetNamespace() {
				t.Errorf("Object namespaces should match. Expected: %s, Got: %s", tt.args.v.GetNamespace(), createdConfigMap.Namespace)
			}
		})
	}
}

func Test_CreateIfNotExists(t *testing.T) {
	type args struct {
		c     client.Client
		v     *corev1.ConfigMap
		until []func(*corev1.ConfigMap) bool
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				c: fake.NewClientBuilder().Build(),
				v: &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "my-configmap",
						Namespace: "default",
					},
					Data: map[string]string{
						"key": "value",
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CreateIfNotExists(tt.args.c, tt.args.v, tt.args.until...); (err != nil) != tt.wantErr {
				t.Errorf("CreateIfNotExists() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Validate the created object
			createdConfigMap := &corev1.ConfigMap{}
			err := tt.args.c.Get(context.TODO(), types.NamespacedName{
				Name:      "my-configmap",
				Namespace: "default",
			}, createdConfigMap)
			if err != nil {
				t.Errorf("Failed to retrieve the created object: %v", err)
			}

			// Verify the object's name and namespace match the expected values
			if createdConfigMap.Name != tt.args.v.GetName() {
				t.Errorf("Object names should match. Expected: %s, Got: %s", tt.args.v.GetName(), createdConfigMap.Name)
			}
			if createdConfigMap.Namespace != tt.args.v.GetNamespace() {
				t.Errorf("Object namespaces should match. Expected: %s, Got: %s", tt.args.v.GetNamespace(), createdConfigMap.Namespace)
			}
		})
	}
}

func Test_CreateWithContext(t *testing.T) {
	type args struct {
		c            client.Client
		ctx          context.Context
		v            *corev1.ConfigMap
		ignoreExists bool
		until        []func(*corev1.ConfigMap) bool
	}

	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				c:   fake.NewClientBuilder().Build(),
				ctx: context.TODO(),
				v: &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "my-configmap",
						Namespace: "default",
					},
					Data: map[string]string{
						"key": "value",
					},
				},
				ignoreExists: false,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := CreateWithContext(tt.args.c, tt.args.ctx, tt.args.v, tt.args.ignoreExists, tt.args.until...); (err != nil) != tt.wantErr {
				t.Errorf("CreateWithContext() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Validate the created object
			createdConfigMap := &corev1.ConfigMap{}
			err := tt.args.c.Get(tt.args.ctx, types.NamespacedName{
				Name:      "my-configmap",
				Namespace: "default",
			}, createdConfigMap)
			if err != nil {
				t.Errorf("Failed to retrieve the created object: %v", err)
			}

			// Verify the object's name and namespace match the expected values
			if createdConfigMap.Name != tt.args.v.GetName() {
				t.Errorf("Object names should match. Expected: %s, Got: %s", tt.args.v.GetName(), createdConfigMap.Name)
			}
			if createdConfigMap.Namespace != tt.args.v.GetNamespace() {
				t.Errorf("Object namespaces should match. Expected: %s, Got: %s", tt.args.v.GetNamespace(), createdConfigMap.Namespace)
			}
		})
	}
}

func Test_DeleteWithContext(t *testing.T) {
	type args struct {
		c   client.Client
		ctx context.Context
		key types.NamespacedName
		v   client.Object
	}

	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				c:   fake.NewClientBuilder().Build(),
				ctx: context.TODO(),
				key: types.NamespacedName{
					Name:      "my-configmap",
					Namespace: "default",
				},
				v: &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "my-configmap",
						Namespace: "default",
					},
					Data: map[string]string{
						"key": "value",
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create the object and save it to Kubernetes
			err := tt.args.c.Create(tt.args.ctx, tt.args.v)
			if err != nil {
				t.Errorf("Failed to create the object: %v", err)
			}

			if err := DeleteWithContext(tt.args.c, tt.args.ctx, tt.args.key, tt.args.v); (err != nil) != tt.wantErr {
				t.Errorf("DeleteWithContext() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Validate if the object is successfully deleted
			deletedConfigMap := &corev1.ConfigMap{}
			err = tt.args.c.Get(tt.args.ctx, types.NamespacedName{
				Name:      "my-configmap",
				Namespace: "default",
			}, deletedConfigMap)
			if tt.wantErr {
				if err == nil || !errors.IsNotFound(err) {
					t.Errorf("Object should be deleted")
				}
			}
		})
	}
}

func Test_Wait(t *testing.T) {
	tests := []struct {
		name      string
		condition wait.ConditionFunc
		wantErr   bool
	}{
		{
			name: "Test case 1",
			condition: func() (bool, error) {
				// Simulate the condition function, returning true indicates the condition is met
				return true, nil
			},
			wantErr: false,
		},
		{
			name: "Test case 2",
			condition: func() (bool, error) {
				timeout := 5 * time.Second
				start := time.Now()
				if time.Since(start) > timeout {
					return false, nil
				}
				// Simulate the condition function, returning true indicates the condition is met
				return true, nil
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Wait(tt.condition); (err != nil) != tt.wantErr {
				t.Errorf("Wait() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_CreateNamespaceIfNotExist(t *testing.T) {
	type args struct {
		c client.Client
	}
	tests := []struct {
		name           string
		args           args
		expectedNsName string
	}{
		{
			name: "Test case 1",
			args: args{
				c: fake.NewClientBuilder().Build(),
			},
			expectedNsName: "expected-namespace",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set global flag
			flags.CreateNamespace = true
			flags.Namespace = tt.expectedNsName

			// Call CreateNamespaceIfNotExist
			CreateNamespaceIfNotExist(tt.args.c)

			// Check if the namespace exists
			actualNamespace := &corev1.Namespace{}
			err := tt.args.c.Get(context.Background(), types.NamespacedName{Name: tt.expectedNsName}, actualNamespace)
			if err != nil {
				t.Fatalf("Expected namespace %s to exist, but got error: %v", tt.expectedNsName, err)
			}
			if actualNamespace.Name != tt.expectedNsName {
				t.Errorf("Expected namespace %s, but got %s", tt.expectedNsName, actualNamespace.Name)
			}
		})
	}
}
