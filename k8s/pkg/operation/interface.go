/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

// Package operation contains the operation logic
package operation

import (
	"context"

	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
)

// PluggableOperation is the interface for the operation
type PluggableOperation interface {
	CreateJob(ctx context.Context, o *v1alpha1.Operation) error
	IsDone() bool
}

// NewPluggableOperation returns a new pluggable operation according to the operation type
func NewPluggableOperation(
	opName string,
	client client.Client,
	clientset *kubernetes.Clientset,
	app *swckkube.Application,
) PluggableOperation {
	switch opName {
	case "assembly":
		return &AssemblyOperation{client, ClientUtils{client}, app, false}
	case "repartition":
		return &RepartitionOperation{client, clientset, ClientUtils{client}, app, false}
	}
	return nil
}
