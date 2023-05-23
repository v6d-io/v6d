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
	"strings"

	"github.com/pkg/errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type Manifests []*unstructured.Unstructured

// ParseManifestToObject parse a kubernetes manifest to an object
func ParseManifestToObject(manifest string) (*unstructured.Unstructured, error) {
	decoder := Deserializer()
	value, _, err := decoder.Decode([]byte(manifest), nil, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to decode resource")
	}
	proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(value)
	if err != nil {
		return nil, errors.Wrap(err, "failed to convert resource to unstructured")
	}
	return &unstructured.Unstructured{Object: proto}, nil
}

// ParseManifestsToObjects parse kubernetes manifests to objects
func ParseManifestsToObjects(manifests []byte) (Manifests, error) {
	// parse the kubernetes yaml file split by "---"
	resources := bytes.Split(manifests, []byte("---"))
	objects := Manifests{}

	for _, f := range resources {
		if string(f) == "\n" || string(f) == "" {
			continue
		}
		obj, err := ParseManifestToObject(string(f))
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse manifest to object")
		}
		objects = append(objects, obj)
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

// ApplyManifestsWithOwnerRef create kubernetes resources from manifests
// and choose one object as the owner of the other specific objects
// Currently, only support to use the kind of manifest as the distinguishing conditions.
// As a result, the ownerKind and refKind should not to be the same. Also, the refKind
// could be more than one and should be separated by comma.
func ApplyManifestsWithOwnerRef(c client.Client, objs []*unstructured.Unstructured,
	ownerKind, refKind string) error {
	newobjs := make([]*unstructured.Unstructured, 0)
	ownerObj := &unstructured.Unstructured{}
	// reorder the objects to deploy the backup job at first
	for _, obj := range objs {
		if obj.GetKind() == ownerKind {
			ownerObj = obj
		} else {
			newobjs = append(newobjs, obj)
		}
	}
	newobjs = append([]*unstructured.Unstructured{ownerObj}, newobjs...)

	ownerRef := metav1.OwnerReference{}
	refKinds := strings.Split(refKind, ",")
	refKindMap := make(map[string]bool)
	for _, kind := range refKinds {
		refKindMap[kind] = true
	}

	for _, obj := range newobjs {
		if _, ok := refKindMap[obj.GetKind()]; ok {
			obj.SetOwnerReferences([]metav1.OwnerReference{ownerRef})
		}
		if err := CreateIfNotExists(c, obj); err != nil {
			return errors.Wrap(err, "failed to create manifest resource")
		}
		if obj.GetKind() == ownerKind {
			if err := c.Get(context.Background(),
				client.ObjectKeyFromObject(obj), ownerObj); err != nil {
				return errors.Wrap(err, "failed to get owner kind object")
			}
			ownerRef = metav1.OwnerReference{
				APIVersion: "batch/v1",
				Kind:       "Job",
				Name:       obj.GetName(),
				UID:        obj.GetUID(),
			}
		}
	}
	return nil
}
