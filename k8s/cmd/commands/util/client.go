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
	"time"

	"k8s.io/client-go/kubernetes"
	defaultscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	corev1 "k8s.io/api/core/v1"
	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

const (
	defaultWaitInternal = 2 * time.Second
	defaultWaitTimeout  = 1 * time.Hour
)

var scheme = runtime.NewScheme()

func init() {
	_ = defaultscheme.AddToScheme(scheme)
	_ = v1alpha1.AddToScheme(scheme)
	_ = apiextensions.AddToScheme(scheme)
}

func Scheme() *runtime.Scheme {
	return scheme
}

func Deserializer() runtime.Decoder {
	return serializer.NewCodecFactory(scheme).UniversalDeserializer()
}

func GetKubernetesConfig() *rest.Config {
	cfg, err := clientcmd.BuildConfigFromFlags("", flags.KubeConfig)
	if err != nil {
		cfg, err = rest.InClusterConfig()
		if err != nil {
			log.Fatal(err, "Failed to resolve the KUBECONFIG")
		}
	}
	return cfg
}

// KubernetesClient return the kubernetes client
func KubernetesClient() client.Client {
	cfg := GetKubernetesConfig()
	client, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		log.Fatal(err, "failed to create the kubernetes API client")
	}
	return client
}

// KubernetesClientset return the kubernetes clientset
func KubernetesClientset() *kubernetes.Clientset {
	cfg := GetKubernetesConfig()
	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		log.Fatal(err, "unable to create REST client")
	}
	return clientset
}

func Create[T client.Object](c client.Client, v T, until ...func(T) bool) error {
	return CreateWithContext(c, context.TODO(), v, false, until...)
}

func CreateIfNotExists[T client.Object](c client.Client, v T, until ...func(T) bool) error {
	return CreateWithContext(c, context.TODO(), v, true, until...)
}

func CreateWithContext[T client.Object](
	c client.Client,
	ctx context.Context,
	v T,
	ignoreExists bool,
	until ...func(T) bool,
) error {
	if ignoreExists {
		err := c.Get(ctx, client.ObjectKeyFromObject(v), v)
		if err == nil {
			return nil
		}
		if apierrors.IsNotFound(err) && v.GetName() != "" {
			err := c.Create(ctx, v)
			if err != nil || len(until) == 0 {
				return err
			}
		}
	} else {
		err := c.Create(ctx, v)
		if err != nil || len(until) == 0 {
			return err
		}
	}

	return Wait(func() (bool, error) {
		err := c.Get(ctx, client.ObjectKeyFromObject(v), v)
		if err != nil {
			if !apierrors.IsNotFound(err) {
				return true, err // early stop if error occurs
			}
		}
		for _, fn := range until {
			if fn == nil || fn(v) {
				return true, nil
			}
		}
		return false, nil
	})
}

func Delete(c client.Client, key types.NamespacedName, v client.Object) error {
	return DeleteWithContext(c, context.TODO(), key, v)
}

func DeleteWithContext(
	c client.Client,
	ctx context.Context,
	key types.NamespacedName,
	v client.Object,
) error {
	_ = c.Get(ctx, key, v)

	if v.GetName() != "" {
		err := c.Delete(ctx, v)
		if err != nil {
			return err
		}
	}

	return Wait(func() (bool, error) {
		if err := c.Get(ctx, key, v); err != nil {
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			log.Error(err, "failed to check the status of deleted object")
			return true, nil // early stop when error occurs, and ignore the error
		}
		return false, nil
	})
}

func Wait(condition wait.ConditionFunc) error {
	return wait.PollImmediate(defaultWaitInternal, defaultWaitTimeout, condition)
}

// CreateNamespaceIfNotExist creates namespace if it does not exist
func CreateNamespaceIfNotExist(c client.Client) {
	if !flags.CreateNamespace {
		return
	}
	namespace := corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: flags.GetDefaultVineyardNamespace(),
		},
	}
	_ = CreateIfNotExists(c, &namespace)
}
