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
	"strings"

	"github.com/pkg/errors"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
)

type ClientUtils struct {
	client.Client
}

// UpdateConfigmap will update the configmap when the assembly operation is done
func (c *ClientUtils) UpdateConfigmap(ctx context.Context, target map[string]bool,
	o *v1alpha1.Operation, prefix string, data *map[string]string,
) error {
	globalObjectList := &v1alpha1.GlobalObjectList{}

	// get all globalobjects which may need to be injected with the assembly job
	if err := c.List(ctx, globalObjectList); err != nil {
		return errors.Wrap(err, "failed to list the global objects")
	}
	// build new global object
	newObjList := []*v1alpha1.GlobalObject{}
	for j := range globalObjectList.Items {
		labels := globalObjectList.Items[j].Labels
		if v, ok := labels[PodNameLabelKey]; ok {
			v = v[len(prefix):strings.LastIndex(v, "-")]
			if target[v] {
				newObjList = append(newObjList, &globalObjectList.Items[j])
			}
		}
	}

	if len(newObjList) != 0 {
		newObjStr := ""
		for i := range newObjList {
			newObjStr = newObjStr + newObjList[i].Name + "."
		}
		newObjStr = newObjStr[:len(newObjStr)-1]

		name := newObjList[0].Labels[labels.VineyardObjectJobLabel]
		namespace := o.Namespace
		// update the configmap
		configmap := &corev1.ConfigMap{}
		err := c.Get(ctx, client.ObjectKey{Name: name, Namespace: namespace}, configmap)
		if err != nil && !apierrors.IsNotFound(err) {
			ctrl.Log.Info("failed to get the configmap")
			return err
		}
		(*data)[name] = newObjStr
		if apierrors.IsNotFound(err) {
			cm := corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: namespace,
				},
				Data: *data,
			}
			if err := c.Create(ctx, &cm); err != nil {
				ctrl.Log.Error(err, "failed to create the configmap")
				return err
			}
		} else {
			// if the configmap exist
			if configmap.Data == nil {
				configmap.Data = map[string]string{}
			}
			configmap.Data[name] = newObjStr
			for k, v := range *data {
				configmap.Data[k] = v
			}
			if err := c.Update(ctx, configmap); err != nil {
				ctrl.Log.Error(err, "failed to update the configmap")
				return err
			}
		}
	}
	return nil
}

func (c *ClientUtils) ResolveRequiredVineyarddSocket(
	ctx context.Context,
	name string, namespace string, objectNamespace string,
) (string, error) {
	vineyardd := &v1alpha1.Vineyardd{}
	namespacedName := types.NamespacedName{
		Name:      name,
		Namespace: namespace,
	}
	if namespacedName.Namespace == "" {
		namespacedName.Namespace = objectNamespace
	}
	if err := c.Get(ctx, namespacedName, vineyardd); err != nil {
		return "", err
	}
	v1alpha1.PreprocessVineyarddSocket(vineyardd)
	return vineyardd.Spec.Vineyard.Socket, nil
}
