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
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
)

func Test_RenderManifestAsObj(t *testing.T) {
	var etcdConfig k8s.EtcdConfig
	opts := &flags.VineyarddOpts
	type args struct {
		path     string
		value    interface{}
		tmplFunc map[string]interface{}
	}
	tests := []struct {
		name    string
		args    args
		want    *unstructured.Unstructured
		wantErr bool
	}{
		{
			name: "Test Case 1",
			args: args{
				path:  "etcd/service.yaml",
				value: *opts,
				tmplFunc: map[string]interface{}{
					"getStorage": func(q resource.Quantity) string {
						return q.String()
					},
					"getEtcdConfig": func() k8s.EtcdConfig {
						return etcdConfig
					},
					"toYaml": func(v interface{}) string {
						bs, err := yaml.Marshal(v)
						if err != nil {
							t.Error(err, "failed to marshal object %v to yaml", v)
							return ""
						}
						return string(bs)
					},
					"indent": func(spaces int, s string) string {
						prefix := strings.Repeat(" ", spaces)
						return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
					},
				},
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Service",
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"etcd_node": "-etcd-0",
						},
						"name":      "-etcd-0",
						"namespace": nil,
					},
					"spec": map[string]interface{}{
						"ports": []interface{}{
							map[string]interface{}{
								"name":       "client",
								"port":       int64(2379),
								"protocol":   "TCP",
								"targetPort": int64(2379),
							},
							map[string]interface{}{
								"name":       "server",
								"port":       int64(2380),
								"protocol":   "TCP",
								"targetPort": int64(2380),
							},
						},
						"selector": map[string]interface{}{
							"app.vineyard.io/role": "etcd",
							"etcd_node":            "-etcd-0",
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RenderManifestAsObj(tt.args.path, tt.args.value, tt.args.tmplFunc)
			if (err != nil) != tt.wantErr {
				t.Errorf("RenderManifestAsObj() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("RenderManifestAsObj() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBuildObjsFromEtcdManifests(t *testing.T) {
	value := &v1alpha1.Vineyardd{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-vineyardd",
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: v1alpha1.VineyarddSpec{
			Replicas:     10,
			EtcdReplicas: 10,
			Service:      v1alpha1.ServiceConfig{Type: "ClusterIP", Port: 8888},
			Vineyard: v1alpha1.VineyardConfig{
				Image:           "test-image",
				ImagePullPolicy: "IfNotPresent",
				SyncCRDs:        true,
				Socket:          "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}",
				ReserveMemory:   false,
				StreamThreshold: 80,
				Spill: v1alpha1.SpillConfig{
					SpillLowerRate: "0.3",
					SpillUpperRate: "0.8",
				},
			},
			PluginImage: v1alpha1.PluginImageConfig{
				BackupImage:              "ghcr.io/v6d-io/v6d/backup-job",
				RecoverImage:             "ghcr.io/v6d-io/v6d/recover-job",
				DaskRepartitionImage:     "ghcr.io/v6d-io/v6d/dask-repartition",
				LocalAssemblyImage:       "ghcr.io/v6d-io/v6d/local-assembly",
				DistributedAssemblyImage: "ghcr.io/v6d-io/v6d/distributed-assembly",
			},
			Metric: v1alpha1.MetricConfig{
				Image:           "vineyardcloudnative/vineyard-grok-exporter:latest",
				ImagePullPolicy: "IfNotPresent",
			},
		},
	}
	var etcdConfig k8s.EtcdConfig
	tmplFunc := map[string]interface{}{
		"getStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
		"toYaml": func(v interface{}) string {
			bs, err := yaml.Marshal(v)
			if err != nil {
				t.Error(err, "failed to marshal object %v to yaml", v)
				return ""
			}
			return string(bs)
		},
		"indent": func(spaces int, s string) string {
			prefix := strings.Repeat(" ", spaces)
			return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
		},
	}
	vineyardd := value
	name := vineyardd.Name
	namespace := vineyardd.Namespace
	replicas := vineyardd.Spec.EtcdReplicas
	image := vineyardd.Spec.Vineyard.Image

	podObjs, svcObjs, err := BuildObjsFromEtcdManifests(&etcdConfig, name, namespace, replicas, image, value, tmplFunc)

	assert.NoError(t, err)
	if len(podObjs) != 10 {
		t.Errorf("Expected %d objects, but got %d", 10, len(podObjs))
	}
	if len(svcObjs) != 10 {
		t.Errorf("Expected %d objects, but got %d", 10, len(svcObjs))
	}

}

func TestBuildObjsFromVineyarddManifests(t *testing.T) {
	files := []string{}
	value := &v1alpha1.Vineyardd{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-vineyardd",
			Namespace: flags.GetDefaultVineyardNamespace(),
		},
		Spec: v1alpha1.VineyarddSpec{
			Replicas:     10,
			EtcdReplicas: 10,
			Service:      v1alpha1.ServiceConfig{Type: "ClusterIP", Port: 8888},
			Vineyard: v1alpha1.VineyardConfig{
				Image:           "test-image",
				ImagePullPolicy: "IfNotPresent",
				SyncCRDs:        true,
				Socket:          "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}",
				ReserveMemory:   false,
				StreamThreshold: 80,
				Spill: v1alpha1.SpillConfig{
					SpillLowerRate: "0.3",
					SpillUpperRate: "0.8",
				},
			},
			PluginImage: v1alpha1.PluginImageConfig{
				BackupImage:              "ghcr.io/v6d-io/v6d/backup-job",
				RecoverImage:             "ghcr.io/v6d-io/v6d/recover-job",
				DaskRepartitionImage:     "ghcr.io/v6d-io/v6d/dask-repartition",
				LocalAssemblyImage:       "ghcr.io/v6d-io/v6d/local-assembly",
				DistributedAssemblyImage: "ghcr.io/v6d-io/v6d/distributed-assembly",
			},
			Metric: v1alpha1.MetricConfig{
				Image:           "vineyardcloudnative/vineyard-grok-exporter:latest",
				ImagePullPolicy: "IfNotPresent",
			},
		},
	}
	var etcdConfig k8s.EtcdConfig
	tmplFunc := map[string]interface{}{
		"getStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
		"toYaml": func(v interface{}) string {
			bs, err := yaml.Marshal(v)
			if err != nil {
				t.Error(err, "failed to marshal object %v to yaml", v)
				return ""
			}
			return string(bs)
		},
		"indent": func(spaces int, s string) string {
			prefix := strings.Repeat(" ", spaces)
			return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
		},
	}

	objs, err := BuildObjsFromVineyarddManifests(files, value, tmplFunc)

	if err != nil {
		t.Fatalf("Failed to build objects from Vineyardd manifests: %v", err)
	}

	if len(objs) != 3 {
		t.Errorf("Expected %d objects, but got %d", 3, len(objs))
	}

}

func TestBuildObjsFromManifests(t *testing.T) {
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

	flags.KubeConfig = kube_config
	c := KubernetesClient()

	// set the vineyardd name and namespace as the vineyard deployment
	backup.Spec.VineyarddName = flags.VineyardDeploymentName
	backup.Spec.VineyarddNamespace = flags.VineyardDeploymentNamespace
	opts := k8s.NewBackupOpts(
		flags.BackupName,
		flags.PVCName,
		flags.BackupOpts.BackupPath,
	)
	backupCfg, _ := opts.BuildCfgForVineyarctl(c, backup)

	tmplFunc := map[string]interface{}{
		"getResourceStorage": func(q resource.Quantity) string {
			return q.String()
		},
		"getBackupConfig": func() k8s.BackupConfig {
			return backupCfg
		},
		"toYaml": func(v interface{}) string {
			bs, err := yaml.Marshal(v)
			if err != nil {
				t.Error(err, "failed to marshal object %v to yaml", v)
				return ""
			}
			return string(bs)
		},
		"indent": func(spaces int, s string) string {
			prefix := strings.Repeat(" ", spaces)
			return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
		},
	}

	objs, err := BuildObjsFromManifests(templateName, backup, tmplFunc)

	assert.NoError(t, err)
	if len(objs) != 4 {
		t.Errorf("Expected %d objects, but got %d", 4, len(objs))
	}

}
