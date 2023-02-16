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
	"bytes"
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func parseManifestsToObjects(manifests []byte) ([]*unstructured.Unstructured, error) {
	// parse the kubernetes yaml file split by "---"
	resources := bytes.Split(manifests, []byte("---"))
	objects := []*unstructured.Unstructured{}

	for _, f := range resources {
		if string(f) == "\n" || string(f) == "" {
			continue
		}

		decode := serializer.NewCodecFactory(CmdScheme).UniversalDeserializer().Decode
		obj, _, err := decode(f, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to decode resource: %v", err)
		}

		proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		if err != nil {
			return nil, fmt.Errorf("failed to convert resource to unstructured: %v", err)
		}

		unstructuredObj := &unstructured.Unstructured{Object: proto}
		objects = append(objects, unstructuredObj)
	}
	return objects, nil
}

// ApplyManifests create kubernetes resouces from manifests
func ApplyManifests(c client.Client, manifests []byte, namespace string) error {
	objs, err := parseManifestsToObjects(manifests)
	if err != nil {
		return err
	}
	for _, obj := range objs {
		// setup the namespace
		if obj.GetNamespace() != "" && namespace != "" {
			obj.SetNamespace(namespace)
		}

		key := client.ObjectKeyFromObject(obj)
		current := &unstructured.Unstructured{}
		current.SetGroupVersionKind(obj.GetObjectKind().GroupVersionKind())
		if err = c.Get(context.TODO(), key, current); err != nil {
			// check whether the unstructed object exist
			if apierrors.IsNotFound(err) {
				if err := c.Create(context.TODO(), obj); err != nil {
					return fmt.Errorf("failed to create resource: %v", err)
				}
			}
		}
	}
	return nil
}

// DeleteManifests delete kubernetes resources from manifests
func DeleteManifests(c client.Client, manifests []byte, namespace string) error {
	objs, err := parseManifestsToObjects(manifests)
	if err != nil {
		return err
	}
	for _, obj := range objs {
		// setup the namespace
		if obj.GetNamespace() != "" {
			obj.SetNamespace(namespace)
		}

		key := client.ObjectKeyFromObject(obj)
		current := &unstructured.Unstructured{}
		current.SetGroupVersionKind(obj.GetObjectKind().GroupVersionKind())

		_ = c.Get(context.TODO(), key, current)
		if current.GetName() != "" {
			if err := c.Delete(context.TODO(), current); err != nil {
				return fmt.Errorf("failed to delete resource: %v", err)
			}
		}
	}
	return nil
}
