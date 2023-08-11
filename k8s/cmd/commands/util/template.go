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
	"strings"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// RenderManifestAsObj renders the manifest as an object
func RenderManifestAsObj(path string, value interface{},
	tmplFunc map[string]interface{},
) (*unstructured.Unstructured, error) {
	obj := &unstructured.Unstructured{}
	manifest, err := templates.ReadFile(path)
	if err != nil {
		return obj, errors.Wrapf(err, "failed to read manifest %s", path)
	}

	_, err = swckkube.LoadTemplate(string(manifest), value, tmplFunc, obj)
	if err != nil && !strings.Contains(err.Error(), "failed load anything from manifests") {
		return obj, errors.Wrapf(err, "failed to render manifest %s", path)
	}
	return obj, nil
}

// BuildObjsFromEtcdManifests builds a list of objects from the etcd template files.
// the template files are under the dir 'k8s/pkg/templates/etcd'
func BuildObjsFromEtcdManifests(etcdConfig *k8s.EtcdConfig, name string,
	namespace string, replicas int, image string, value interface{},
	tmplFunc map[string]interface{},
) ([]*unstructured.Unstructured, []*unstructured.Unstructured, error) {
	podObjs := []*unstructured.Unstructured{}
	svcObjs := []*unstructured.Unstructured{}

	etcdManifests, err := templates.GetFilesRecursive("etcd")
	if err != nil {
		return podObjs, svcObjs, errors.Wrap(err, "failed to get etcd manifests")
	}
	// set up the etcd config
	*etcdConfig = k8s.NewEtcdConfig(name, namespace, replicas, image)

	for i := 0; i < replicas; i++ {
		etcdConfig.Rank = i
		for _, ef := range etcdManifests {
			obj, err := RenderManifestAsObj(ef, value, tmplFunc)
			if err != nil {
				return podObjs, svcObjs, err
			}

			if ef == "etcd/service.yaml" && obj.GetName() != "" {
				svcObjs = append(svcObjs, obj)
			}

			if ef == "etcd/etcd.yaml" && obj.GetName() != "" {
				podObjs = append(podObjs, obj)
			}
		}
	}
	return podObjs, svcObjs, nil
}

// BuildObjsFromVineyarddManifests builds a list of objects from the
// vineyardd template files.
func BuildObjsFromVineyarddManifests(files []string, value interface{},
	tmplFunc map[string]interface{},
) ([]*unstructured.Unstructured, error) {
	objs := []*unstructured.Unstructured{}

	fileExists := make(map[string]bool)
	for _, f := range files {
		fileExists[f] = true
	}

	vineyardManifests, err := templates.GetFilesRecursive("vineyardd")
	if err != nil {
		return objs, errors.Wrap(err, "failed to get vineyard manifests")
	}
	for _, vf := range vineyardManifests {
		if _, ok := fileExists[vf]; len(files) == 0 || ok {
			obj, err := RenderManifestAsObj(vf, value, tmplFunc)
			if err != nil {
				return objs, err
			}

			if obj.GetName() != "" {
				objs = append(objs, obj)
			}
		}
	}
	return objs, nil
}

// BuildObjsFromManifests builds a list of objects from the
// the standard template files such as backup and recover
func BuildObjsFromManifests(templateName string, value interface{},
	tmplFunc map[string]interface{},
) ([]*unstructured.Unstructured, error) {
	objs := []*unstructured.Unstructured{}

	manifests, err := templates.GetFilesRecursive(templateName)
	if err != nil {
		return objs, errors.Wrap(err, "failed to get backup manifests")
	}
	for _, m := range manifests {
		obj, err := RenderManifestAsObj(m, value, tmplFunc)
		if err != nil {
			return objs, err
		}

		if flags.PVCName != "" && (obj.GetKind() == "PersistentVolumeClaim" || obj.GetKind() == "PersistentVolume") {
			continue
		}
		if obj.GetName() != "" {
			objs = append(objs, obj)
		}
	}
	return objs, nil
}
