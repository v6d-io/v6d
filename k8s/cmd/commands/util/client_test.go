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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func Scheme_copy() *runtime.Scheme {
	return scheme
}
func TestScheme(t *testing.T) {
	var expectedScheme = Scheme_copy()

	// 调用 Scheme 函数获取实际的 scheme 对象
	var actualScheme = Scheme()

	// 验证返回的 scheme 对象是否与预期相同
	if actualScheme != expectedScheme {
		t.Errorf("Expected scheme: %v, got: %v", expectedScheme, actualScheme)
	}
}

func TestDeserializer(t *testing.T) {
	// 调用 Deserializer 函数获取实际的 decoder 对象
	decoder := Deserializer()

	// 验证返回的 decoder 对象是否为非空值
	if decoder == nil {
		t.Error("Decoder should not be nil")
	}

	// 进一步验证 decoder 对象的类型
	_, ok := decoder.(runtime.Decoder)
	if !ok {
		t.Error("Decoder should be of type runtime.Decoder")
	}
}

func TestGetKubernetesConfig(t *testing.T) {
	// 模拟flags.KubeConfig的值
	flags.KubeConfig = "/home/zhuyi/.kube/config"

	// 调用getKubernetesConfig函数
	cfg := GetKubernetesConfig()

	// 验证是否成功地获取了*rest.Config
	if cfg == nil {
		t.Errorf("Expected *rest.Config, but got nil")
	}
}

func TestKubernetesClient(t *testing.T) {
	// 模拟flags.KubeConfig的值
	flags.KubeConfig = "/home/zhuyi/.kube/config"

	// 调用KubernetesClient函数
	kubeClient := KubernetesClient()

	// 验证是否成功地获取了client.Client
	if kubeClient == nil {
		t.Errorf("Expected client.Client, but got nil")
	}
}

func TestKubernetesClientset(t *testing.T) {
	// 模拟flags.KubeConfig的值
	flags.KubeConfig = "/home/zhuyi/.kube/config"

	// 调用KubernetesClientset函数
	clientset := KubernetesClientset()

	// 验证是否成功地获取了*kubernetes.Clientset
	if clientset == nil {
		t.Errorf("Expected *kubernetes.Clientset, but got nil")
	}
}

func TestCreate(t *testing.T) {
	// 创建一个 Fake Client
	fakeClient := fake.NewFakeClient()

	// 创建一个需要创建的对象（示例中使用 ConfigMap）
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-configmap",
			Namespace: "default",
		},
		Data: map[string]string{
			"key": "value",
		},
	}

	// 调用 Create 函数创建对象
	err := Create(fakeClient, configMap)
	assert.NoError(t, err, "Create should not return an error")

	// 验证创建的对象是否存在
	createdConfigMap := &corev1.ConfigMap{}
	err = fakeClient.Get(context.TODO(), types.NamespacedName{
		Name:      "my-configmap",
		Namespace: "default",
	}, createdConfigMap)
	assert.NoError(t, err, "Failed to retrieve the created object")

	// 验证创建的对象是否与预期一致
	assert.Equal(t, configMap.Name, createdConfigMap.Name, "Object names should match")
	assert.Equal(t, configMap.Namespace, createdConfigMap.Namespace, "Object namespaces should match")
}

func TestCreateIfNotExists(t *testing.T) {
	// 创建一个 Fake Client
	fakeClient := fake.NewFakeClient()

	// 创建一个需要创建的对象（示例中使用 ConfigMap）
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-configmap",
			Namespace: "default",
		},
		Data: map[string]string{
			"key": "value",
		},
	}

	// 调用 CreateIfNotExists 函数创建对象
	err := CreateIfNotExists(fakeClient, configMap)
	assert.NoError(t, err, "CreateIfNotExists should not return an error")

	// 验证创建的对象是否存在
	createdConfigMap := &corev1.ConfigMap{}
	err = fakeClient.Get(context.TODO(), types.NamespacedName{
		Name:      "my-configmap",
		Namespace: "default",
	}, createdConfigMap)
	assert.NoError(t, err, "Failed to retrieve the created object")

	// 验证创建的对象是否与预期一致
	assert.Equal(t, configMap.Name, createdConfigMap.Name, "Object names should match")
	assert.Equal(t, configMap.Namespace, createdConfigMap.Namespace, "Object namespaces should match")
}

func TestCreateWithContext(t *testing.T) {
	// 创建一个 Fake Client
	fakeClient := fake.NewFakeClient()

	// 创建一个需要创建的对象（示例中使用 ConfigMap）
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-configmap",
			Namespace: "default",
		},
		Data: map[string]string{
			"key": "value",
		},
	}

	// 调用 CreateWithContext 函数创建对象（ignoreExists 为 false）
	err := CreateWithContext(fakeClient, context.TODO(), configMap, false)
	assert.NoError(t, err, "CreateWithContext should not return an error")

	// 验证创建的对象是否存在
	createdConfigMap := &corev1.ConfigMap{}
	err = fakeClient.Get(context.TODO(), types.NamespacedName{
		Name:      "my-configmap",
		Namespace: "default",
	}, createdConfigMap)
	assert.NoError(t, err, "Failed to retrieve the created object")

	// 验证创建的对象是否与预期一致
	assert.Equal(t, configMap.Name, createdConfigMap.Name, "Object names should match")
	assert.Equal(t, configMap.Namespace, createdConfigMap.Namespace, "Object namespaces should match")
}

func TestDelete(t *testing.T) {
	// 创建一个 Fake Client
	fakeClient := fake.NewFakeClient()

	// 创建一个需要删除的对象（示例中使用 ConfigMap）
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-configmap",
			Namespace: "default",
		},
		Data: map[string]string{
			"key": "value",
		},
	}

	// 创建对象并保存到 Kubernetes 中
	err := fakeClient.Create(context.TODO(), configMap)
	assert.NoError(t, err, "Failed to create the object")

	// 调用 Delete 函数删除对象
	err = Delete(fakeClient, types.NamespacedName{Name: "my-configmap", Namespace: "default"}, configMap)
	assert.NoError(t, err, "Delete should not return an error")

	// 验证对象是否被成功删除
	deletedConfigMap := &corev1.ConfigMap{}
	err = fakeClient.Get(context.TODO(), types.NamespacedName{
		Name:      "my-configmap",
		Namespace: "default",
	}, deletedConfigMap)
	assert.True(t, errors.IsNotFound(err), "Object should be deleted")
}

func TestDeleteWithContext(t *testing.T) {
	// 创建一个 Fake Client
	fakeClient := fake.NewFakeClient()

	// 创建一个需要删除的对象（示例中使用 ConfigMap）
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-configmap",
			Namespace: "default",
		},
		Data: map[string]string{
			"key": "value",
		},
	}

	// 创建对象并保存到 Kubernetes 中
	err := fakeClient.Create(context.TODO(), configMap)
	assert.NoError(t, err, "Failed to create the object")

	// 调用 DeleteWithContext 函数删除对象
	err = DeleteWithContext(fakeClient, context.TODO(), types.NamespacedName{Name: "my-configmap", Namespace: "default"}, configMap)
	assert.NoError(t, err, "DeleteWithContext should not return an error")

	// 验证对象是否被成功删除
	deletedConfigMap := &corev1.ConfigMap{}
	err = fakeClient.Get(context.TODO(), types.NamespacedName{
		Name:      "my-configmap",
		Namespace: "default",
	}, deletedConfigMap)
	assert.True(t, errors.IsNotFound(err), "Object should be deleted")
}

func TestWait(t *testing.T) {
	// 设置一个简单的条件函数，用于测试等待
	condition := func() (bool, error) {
		// 模拟条件函数，返回 true 表示满足条件
		return true, nil
	}

	// 调用 Wait 函数进行等待
	err := Wait(condition)
	assert.NoError(t, err, "Wait should not return an error")

	// 可以使用更复杂的条件函数和等待时间进行测试
	// 例如，使用 time.After 设置一个超时时间，如果超时仍未满足条件，则测试失败
	timeout := 5 * time.Second
	start := time.Now()
	condition = func() (bool, error) {
		if time.Since(start) > timeout {
			return false, nil
		}
		// 模拟条件函数，返回 true 表示满足条件
		return true, nil
	}

	err = Wait(condition)
	assert.NoError(t, err, "Wait should not return an error")
}

func TestCreateNamespaceIfNotExist(t *testing.T) {
	// 创建一个带有预期 namespace 的模拟 client
	expectedNamespace := "expected-namespace"
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: expectedNamespace,
		},
	}
	c := fake.NewFakeClientWithScheme(scheme, namespace)

	// 设置全局 flag
	flags.CreateNamespace = true
	flags.Namespace = expectedNamespace

	// 调用 CreateNamespaceIfNotExist
	CreateNamespaceIfNotExist(c)

	// 检查 namespace 是否存在
	actualNamespace := &corev1.Namespace{}
	err := c.Get(context.Background(), types.NamespacedName{Name: expectedNamespace}, actualNamespace)
	if err != nil {
		t.Fatalf("Expected namespace %s to exist, but got error: %v", expectedNamespace, err)
	}
	if actualNamespace.Name != expectedNamespace {
		t.Errorf("Expected namespace %s, but got %s", expectedNamespace, actualNamespace.Name)
	}
}
