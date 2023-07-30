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
	_ "embed"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRenderManifestAsObj(t *testing.T) {
	path := "etcd/service.yaml"
	opts := &flags.VineyarddOpts
	value := &v1alpha1.Vineyardd{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.VineyarddName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: *opts,
	}
	var etcdConfig k8s.EtcdConfig
	tmplFunc := map[string]interface{}{
		"getStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
	}

	obj, err := RenderManifestAsObj(path, value, tmplFunc)

	assert.NoError(t, err)
	assert.NotNil(t, obj)

}

func TestBuildObjsFromEtcdManifests(t *testing.T) {

	opts := &flags.VineyarddOpts
	value := &v1alpha1.Vineyardd{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.VineyarddName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: *opts,
	}
	var etcdConfig k8s.EtcdConfig
	tmplFunc := map[string]interface{}{
		"getStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
	}
	vineyardd := value
	name := vineyardd.Name
	namespace := vineyardd.Namespace
	replicas := vineyardd.Spec.EtcdReplicas
	image := vineyardd.Spec.Vineyard.Image

	_, _, err := BuildObjsFromEtcdManifests(&etcdConfig, name, namespace, replicas, image, value, tmplFunc)

	assert.NoError(t, err)

}

func TestBuildObjsFromVineyarddManifests(t *testing.T) {
	files := []string{"etcd/service.yaml"}
	opts := &flags.VineyarddOpts
	value := &v1alpha1.Vineyardd{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.VineyarddName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: *opts,
	}
	var etcdConfig k8s.EtcdConfig
	tmplFunc := map[string]interface{}{
		"getStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
	}

	objs, err := BuildObjsFromVineyarddManifests(files, value, tmplFunc)

	if err != nil {
		t.Fatalf("Failed to build objects from Vineyardd manifests: %v", err)
	}

	if len(objs) != 0 {
		t.Errorf("Expected %d objects, but got %d", 0, len(objs))
	}

}

/*func TestBuildObjsFromManifests(t *testing.T) {
	templateName := "backup"

	backup := &v1alpha1.Backup{
		TypeMeta: metav1.TypeMeta{},
		ObjectMeta: metav1.ObjectMeta{
			Name:        "vineyard-backup",
			Namespace:   "",
			UID:         "",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
			CreationTimestamp: metav1.Time{
				Time: time.Date(1, 1, 1, 0, 0, 0, 0, time.UTC),
			},
			DeletionTimestamp: nil,
			OwnerReferences:   []metav1.OwnerReference{},
			Finalizers:        []string{},
			//ClusterName:       "",
			ManagedFields: []metav1.ManagedFieldsEntry{},
		},
	}

	flags.KubeConfig = "/home/zhuyi/.kube/config"
	c := KubernetesClient()

	useVineyardScheduler := false
	path := flags.BackupOpts.BackupPath
	// set the vineyardd name and namespace as the vineyard deployment
	backup.Spec.VineyarddName = flags.VineyardDeploymentName
	backup.Spec.VineyarddNamespace = flags.VineyardDeploymentNamespace
	backupCfg, err := k8s.buildBackupCfg(c, flags.BackupName, backup, path, useVineyardScheduler)

	tmplFunc := map[string]interface{}{
		"getResourceStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getBackupConfig": func() k8s.BackupConfig {
			return backupCfg
		},
	}

	objs, err := BuildObjsFromManifests(templateName, backup, tmplFunc)

	assert.NoError(t, err)
	assert.NotEmpty(t, objs)

}*/
