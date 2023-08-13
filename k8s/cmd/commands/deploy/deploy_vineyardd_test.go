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
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	corev1 "k8s.io/api/core/v1"

	//corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDeployVineyarddCmd(t *testing.T) {
	test := struct {
		name                 string
		vineyardReplicas     int
		etcdReplicas         int
		expectedImage        string
		expectedCpu          string
		expectedMemery       string
		expectedService_port int
		expectedService_type string
	}{
		name:                 "test replicas",
		vineyardReplicas:     3,
		etcdReplicas:         1,
		expectedImage:        "vineyardcloudnative/vineyardd:alpine-latest",
		expectedCpu:          "",
		expectedMemery:       "",
		expectedService_port: 9600,
		expectedService_type: "ClusterIP",
	}
	t.Run(test.name, func(t *testing.T) {
		// set the flags
		flags.Namespace = "vineyard-system"
		//flags.KubeConfig = os.Getenv("HOME") + "/.kube/config"
		flags.KubeConfig = "/tmp/e2e-k8s.config"
		flags.VineyarddOpts.Replicas = 3
		flags.VineyarddOpts.EtcdReplicas = 1
		flags.VineyarddOpts.Vineyard.Image = "vineyardcloudnative/vineyardd:alpine-latest"
		flags.VineyarddOpts.Vineyard.CPU = ""
		flags.VineyarddOpts.Vineyard.Memory = ""
		flags.VineyarddOpts.Service.Port = 9600
		flags.VineyarddOpts.Service.Type = "ClusterIP"
		flags.VineyarddOpts.Vineyard.Size = "256Mi"
		deployVineyarddCmd.Run(deployVineyarddCmd, []string{})
		time.Sleep(1 * time.Second)
		// get the replicas of etcd and vineyardd
		k8sclient := util.KubernetesClient()
		vineyardPods := corev1.PodList{}
		vineyarddOpts := []client.ListOption{
			client.InNamespace(flags.Namespace),
			client.MatchingLabels{
				"app.vineyard.io/name": flags.VineyarddName,
				"app.vineyard.io/role": "vineyardd",
			},
		}
		err := k8sclient.List(context.Background(), &vineyardPods, vineyarddOpts...)
		if err != nil {
			t.Errorf("list vineyardd pods error: %v", err)
		}

		etcdPod := corev1.PodList{}
		etcdOpts := []client.ListOption{
			client.InNamespace(flags.Namespace),
			client.MatchingLabels{
				"app.vineyard.io/name": flags.VineyarddName,
				"app.vineyard.io/role": "etcd",
			},
		}
		err = k8sclient.List(context.Background(), &etcdPod, etcdOpts...)
		if err != nil {
			t.Errorf("list etcd pods error: %v", err)
		}

		for _, pod := range vineyardPods.Items {
			for _, container := range pod.Spec.Containers {
				if container.Image != test.expectedImage {
					t.Errorf("Pod %s in namespace %s uses the image %s, expected image %s\n", pod.Name, pod.Namespace, container.Image, test.expectedImage)
				}
				if cpuRequest, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
					if cpuRequest.String() != test.expectedCpu {
						t.Errorf("Pod %s in namespace %s has cpu request %s, expected cpu request %s\n", pod.Name, pod.Namespace, cpuRequest.String(), test.expectedCpu)
					}
				}
				if memoryRequest, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
					if memoryRequest.String() != test.expectedMemery {
						t.Errorf("Pod %s in namespace %s has memory request %s, expected memory request %s\n", pod.Name, pod.Namespace, memoryRequest.String(), test.expectedMemery)
					}
				}
			}
		}
		if len(vineyardPods.Items) != test.vineyardReplicas {
			t.Errorf("vineyardd replicas want: %d, got: %d", test.vineyardReplicas, len(vineyardPods.Items))
		}

		for _, pod := range etcdPod.Items {
			for _, container := range pod.Spec.Containers {
				if container.Image != test.expectedImage {
					t.Errorf("Pod %s in namespace %s uses the image %s, expected image %s\n", pod.Name, pod.Namespace, container.Image, test.expectedImage)
				}
				if cpuRequest, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
					if cpuRequest.String() != test.expectedCpu {
						t.Errorf("Pod %s in namespace %s has cpu request %s, expected cpu request %s\n", pod.Name, pod.Namespace, cpuRequest.String(), test.expectedCpu)
					}
				}
				if memoryRequest, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
					if memoryRequest.String() != test.expectedMemery {
						t.Errorf("Pod %s in namespace %s has memory request %s, expected memory request %s\n", pod.Name, pod.Namespace, memoryRequest.String(), test.expectedMemery)
					}
				}
			}
		}
		if len(etcdPod.Items) != test.etcdReplicas {
			t.Errorf("etcd replicas want: %d, got: %d", test.etcdReplicas, len(etcdPod.Items))
		}

		// get the service object
		svcList := corev1.ServiceList{}
		err = k8sclient.List(context.Background(), &svcList, client.MatchingLabels{"app.vineyard.io/name": "vineyardd-sample"})
		if err != nil {
			t.Errorf("list services error: %v", err)
		}
		for _, svc := range svcList.Items {
			if svc.Spec.Ports[0].Port != int32(test.expectedService_port) {
				t.Errorf("Service %s in namespace %s uses the port %d, expected port %d\n", svc.Name, svc.Namespace, svc.Spec.Ports[0].Port, test.expectedService_port)
			}
			if string(svc.Spec.Type) != test.expectedService_type {
				t.Errorf("Service %s in namespace %s uses the type %s, expected type %s\n", svc.Name, svc.Namespace, string(svc.Spec.Type), test.expectedService_type)
			}
		}

	})
}

func TestBuildVineyard(t *testing.T) {
	// set the different flags
	flags.VineyarddName = "test-vineyardd"
	flags.VineyarddOpts.Replicas = 10
	flags.VineyarddOpts.EtcdReplicas = 10
	flags.VineyarddOpts.Service.Port = 8888
	flags.VineyarddOpts.Vineyard.Image = "test-image"

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

func TestBuildVineyardManifestFromInput(t *testing.T) {
	// set the different flags
	flags.VineyarddName = "test-vineyardd"
	flags.VineyarddOpts.Replicas = 10
	flags.VineyarddOpts.EtcdReplicas = 10
	flags.VineyarddOpts.Service.Port = 8888
	flags.VineyarddOpts.Vineyard.Image = "test-image"

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

func TestBuildVineyardManifestFromFile(t *testing.T) {
	// set the flags
	flags.Namespace = "vineyard-system"
	//flags.VineyarddFile = os.Getenv("HOME") + "/v6d/k8s/test/e2e/vineyardd.yaml"
	currentDir, _ := os.Getwd()
	flags.VineyarddFile = filepath.Join(currentDir, "..", "..", "..", "test/e2e/vineyardd.yaml")

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
						Image:           "localhost:5001/vineyardd:alpine-latest",
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
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BuildVineyardManifestFromFile() = %v, want %v", got, tt.want)
			}
		})
	}
}
