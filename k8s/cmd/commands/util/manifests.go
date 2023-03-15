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

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type Manifests []*unstructured.Unstructured

// ParseManifestsToObjects parse kubernetes manifests to objects
func ParseManifestsToObjects(manifests []byte) (Manifests, error) {
	// parse the kubernetes yaml file split by "---"
	resources := bytes.Split(manifests, []byte("---"))
	objects := Manifests{}

	decoder := Deserializer()
	for _, f := range resources {
		if string(f) == "\n" || string(f) == "" {
			continue
		}
		value, _, err := decoder.Decode(f, nil, nil)
		if err != nil {
			return nil, errors.Wrap(err, "failed to decode resource")
		}
		proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(value)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert resource to unstructured")
		}
		object := &unstructured.Unstructured{Object: proto}
		objects = append(objects, object)
	}
	return objects, nil
}

// ApplyManifests create kubernetes resources from manifests
func ApplyManifests(c client.Client, manifests Manifests, namespace string) error {
	for _, object := range manifests {
		// setup the namespace
		if object.GetNamespace() != "" && namespace != "" {
			object.SetNamespace(namespace)
		}
		if err := CreateIfNotExists(c, object); err != nil {
			return errors.Wrap(err, "Failed to create manifest resource")
		}
	}
	return nil
}

// DeleteManifests delete kubernetes resources from manifests
func DeleteManifests(c client.Client, manifests Manifests, namespace string) error {
	for _, object := range manifests {
		// setup the namespace
		if object.GetNamespace() != "" && namespace != "" {
			object.SetNamespace(namespace)
		}

		key := client.ObjectKeyFromObject(object)
		current := &unstructured.Unstructured{}
		current.SetGroupVersionKind(object.GetObjectKind().GroupVersionKind())
		if err := Delete(c, key, current); err != nil {
			return errors.Wrap(err, "failed to delete manifest resource")
		}
	}
	return nil
}
