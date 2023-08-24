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
package deploy

import (
	"fmt"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

func TestBuildVineyard_third(t *testing.T) {
	// set the different flags
	flags.VineyarddName = "test-vineyardd-1"
	flags.VineyarddOpts.Replicas = 10
	flags.VineyarddOpts.EtcdReplicas = 10
	flags.VineyarddOpts.Service.Port = 8888
	flags.VineyarddOpts.Vineyard.Image = "test-image-1"

	tests := []struct {
		name    string
		want    *v1alpha1.Vineyardd
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case",
			want: &v1alpha1.Vineyardd{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-vineyardd-1",
					Namespace: flags.GetDefaultVineyardNamespace(),
				},
				Spec: v1alpha1.VineyarddSpec{
					Replicas:     10,
					EtcdReplicas: 10,
					Service:      v1alpha1.ServiceConfig{Type: "ClusterIP", Port: 8888},
					Vineyard: v1alpha1.VineyardConfig{
						Image:           "test-image-1",
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
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildVineyard()
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildVineyard() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BuildVineyard() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBuildVineyardManifestFromInput_third(t *testing.T) {
	// set the different flags
	flags.VineyarddName = "test-vineyardd-2"
	flags.VineyarddOpts.Replicas = 10
	flags.VineyarddOpts.EtcdReplicas = 10
	flags.VineyarddOpts.Service.Port = 8888
	flags.VineyarddOpts.Vineyard.Image = "test-image-2"

	tests := []struct {
		name    string
		want    *v1alpha1.Vineyardd
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Vineyardd{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-vineyardd-2",
					Namespace: flags.GetDefaultVineyardNamespace(),
				},
				Spec: v1alpha1.VineyarddSpec{
					Replicas:     10,
					EtcdReplicas: 10,
					Service:      v1alpha1.ServiceConfig{Type: "ClusterIP", Port: 8888},
					Vineyard: v1alpha1.VineyardConfig{
						Image:           "test-image-2",
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
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildVineyardManifestFromInput()
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildVineyardManifestFromInput() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BuildVineyardManifestFromInput() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBuildVineyardManifestFromFile_first(t *testing.T) {
	// set the flags
	flags.Namespace = vineyard_default_namespace
	flags.VineyarddFile = "../../../test/e2e/vineyardd.yaml"

	tests := []struct {
		name    string
		want    *v1alpha1.Vineyardd
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Vineyardd{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Vineyardd",
					APIVersion: "k8s.v6d.io/v1alpha1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "vineyardd-sample",
					Namespace: "vineyard-system",
				},
				Spec: v1alpha1.VineyarddSpec{
					Replicas:     3,
					EtcdReplicas: 0,
					Service:      v1alpha1.ServiceConfig{Type: "ClusterIP", Port: 9600},
					Vineyard: v1alpha1.VineyardConfig{
						Image:           "localhost:5001/vineyardd:latest",
						ImagePullPolicy: "IfNotPresent",
						SyncCRDs:        false,
						ReserveMemory:   false,
						StreamThreshold: 0,
						Spill:           v1alpha1.SpillConfig{},
					},
					PluginImage: v1alpha1.PluginImageConfig{
						BackupImage:              "localhost:5001/backup-job",
						RecoverImage:             "localhost:5001/recover-job",
						DaskRepartitionImage:     "localhost:5001/dask-repartition",
						LocalAssemblyImage:       "localhost:5001/local-assembly",
						DistributedAssemblyImage: "localhost:5001/distributed-assembly",
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildVineyardManifestFromFile()
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildVineyardManifestFromFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			gotStr := fmt.Sprintf("%v", got)
			wantStr := fmt.Sprintf("%v", tt.want)
			if gotStr != wantStr {
				t.Errorf("BuildVineyardManifestFromFile() = %v, want %v", gotStr, wantStr)
			}
		})
	}
}
