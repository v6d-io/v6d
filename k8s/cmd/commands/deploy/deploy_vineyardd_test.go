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
	"reflect"
	"testing"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"

	//corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

/*func TestDeployVineyarddCmd(t *testing.T) {
	testReplicas := struct {
		name             string
		vineyardReplicas int
		etcdReplicas     int
	}{
		name:             "test replicas",
		vineyardReplicas: 1,
		etcdReplicas:     1,
	}
	t.Run(testReplicas.name, func(t *testing.T) {
		// set the flags
		//flags.KubeConfig = "/tmp/e2e-k8s.config"
		flags.Namespace = "vineyard-system"
		flags.KubeConfig = "/home/zhuyi/.kube/config"
		flags.VineyarddOpts.Replicas = 1
		flags.VineyarddOpts.EtcdReplicas = 1
		deployVineyarddCmd.Run(deployVineyarddCmd, []string{})
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

		if len(vineyardPods.Items) != testReplicas.vineyardReplicas {
			t.Errorf("vineyardd replicas want: %d, got: %d", testReplicas.vineyardReplicas, len(vineyardPods.Items))
		}

		if len(etcdPod.Items) != testReplicas.etcdReplicas {
			t.Errorf("etcd replicas want: %d, got: %d", testReplicas.etcdReplicas, len(etcdPod.Items))
		}

	})
}*/

func TestDeployVineyarddCmd(t *testing.T) {
	testReplicas := struct {
		name                   string
		vineyardReplicas       int
		etcdReplicas           int
		expectedImage          string
		expectedCpu            string
		expectedMemery         string
		expectedService_port   int
		expectedService_type   string
		expectedSolume_pvcname string
	}{
		name:                   "test replicas",
		vineyardReplicas:       1,
		etcdReplicas:           1,
		expectedImage:          "vineyardcloudnative/vineyardd:latest",
		expectedCpu:            "",
		expectedMemery:         "",
		expectedService_port:   9600,
		expectedService_type:   "ClusterIP",
		expectedSolume_pvcname: "",
	}
	t.Run(testReplicas.name, func(t *testing.T) {
		// set the flags
		//flags.KubeConfig = "/tmp/e2e-k8s.config"
		flags.Namespace = "vineyard-system"
		flags.KubeConfig = "/home/zhuyi/.kube/config"
		flags.VineyarddOpts.Replicas = 1
		flags.VineyarddOpts.EtcdReplicas = 1
		flags.VineyarddOpts.Vineyard.Image = "vineyardcloudnative/vineyardd:latest"
		flags.VineyarddOpts.Vineyard.CPU = ""
		flags.VineyarddOpts.Vineyard.Memory = ""
		flags.VineyarddOpts.Service.Port = 9600
		flags.VineyarddOpts.Service.Type = "ClusterIP"
		flags.VineyarddOpts.Volume.PvcName = ""
		flags.VineyarddOpts.Vineyard.Size = "256Mi"
		deployVineyarddCmd.Run(deployVineyarddCmd, []string{})
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
				if container.Image != testReplicas.expectedImage {
					t.Errorf("Pod %s in namespace %s uses the image %s, expected image %s\n", pod.Name, pod.Namespace, container.Image, testReplicas.expectedImage)
				}
				if cpuRequest, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
					if cpuRequest.String() != testReplicas.expectedCpu {
						t.Errorf("Pod %s in namespace %s has cpu request %s, expected cpu request %s\n", pod.Name, pod.Namespace, cpuRequest.String(), testReplicas.expectedCpu)
					}
				}
				if memoryRequest, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
					if memoryRequest.String() != testReplicas.expectedMemery {
						t.Errorf("Pod %s in namespace %s has memory request %s, expected memory request %s\n", pod.Name, pod.Namespace, memoryRequest.String(), testReplicas.expectedMemery)
					}
				}
			}
		}
		if len(vineyardPods.Items) != testReplicas.vineyardReplicas {
			t.Errorf("vineyardd replicas want: %d, got: %d", testReplicas.vineyardReplicas, len(vineyardPods.Items))
		}

		for _, pod := range etcdPod.Items {
			for _, container := range pod.Spec.Containers {
				if container.Image != testReplicas.expectedImage {
					t.Errorf("Pod %s in namespace %s uses the image %s, expected image %s\n", pod.Name, pod.Namespace, container.Image, testReplicas.expectedImage)
				}
				if cpuRequest, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
					if cpuRequest.String() != testReplicas.expectedCpu {
						t.Errorf("Pod %s in namespace %s has cpu request %s, expected cpu request %s\n", pod.Name, pod.Namespace, cpuRequest.String(), testReplicas.expectedCpu)
					}
				}
				if memoryRequest, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
					if memoryRequest.String() != testReplicas.expectedMemery {
						t.Errorf("Pod %s in namespace %s has memory request %s, expected memory request %s\n", pod.Name, pod.Namespace, memoryRequest.String(), testReplicas.expectedMemery)
					}
				}
			}
		}
		if len(etcdPod.Items) != testReplicas.etcdReplicas {
			t.Errorf("etcd replicas want: %d, got: %d", testReplicas.etcdReplicas, len(etcdPod.Items))
		}

		// 获取服务对象
		svcList := corev1.ServiceList{}
		err = k8sclient.List(context.Background(), &svcList, client.MatchingLabels{"app.vineyard.io/name": "vineyardd-sample"})
		if err != nil {
			t.Errorf("list services error: %v", err)
		}
		for _, svc := range svcList.Items {
			if svc.Spec.Ports[0].Port != int32(testReplicas.expectedService_port) {
				t.Errorf("Service %s in namespace %s uses the port %d, expected port %d\n", svc.Name, svc.Namespace, svc.Spec.Ports[0].Port, testReplicas.expectedService_port)
			}
			if string(svc.Spec.Type) != testReplicas.expectedService_type {
				t.Errorf("Service %s in namespace %s uses the type %s, expected type %s\n", svc.Name, svc.Namespace, string(svc.Spec.Type), testReplicas.expectedService_type)
			}
		}

		// 获取 PVC 对象
		pvcList := corev1.PersistentVolumeClaimList{}
		err = k8sclient.List(context.Background(), &pvcList, client.InNamespace(flags.Namespace))
		//fmt.Println(&pvcList)
		if err != nil {
			t.Errorf("list PVCs error: %v", err)
		}
		for _, pvc := range pvcList.Items {
			if pvc.Name != testReplicas.expectedSolume_pvcname {
				t.Errorf("PVC %s in namespace %s is not expected, expected pvc name %s\n", pvc.Name, pvc.Namespace, testReplicas.expectedSolume_pvcname)
			}
		}
	})
}

func TestBuildVineyard(t *testing.T) {
	opts := &flags.VineyarddOpts

	tests := []struct {
		name    string
		want    *v1alpha1.Vineyardd
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Vineyardd{
				ObjectMeta: metav1.ObjectMeta{
					Name:      flags.VineyarddName,
					Namespace: flags.GetDefaultVineyardNamespace(),
				},
				Spec: *opts,
			}, // 指定预期的 *cobra.Command 值
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
	opts := &flags.VineyarddOpts
	tests := []struct {
		name    string
		want    *v1alpha1.Vineyardd
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Vineyardd{
				ObjectMeta: metav1.ObjectMeta{
					Name:      flags.VineyarddName,
					Namespace: flags.GetDefaultVineyardNamespace(),
				},
				Spec: *opts,
			}, // 指定预期的 *cobra.Command 值
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
	tests := []struct {
		name    string
		want    *v1alpha1.Vineyardd
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Vineyardd{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "vineyard-system",
				},
				Spec: v1alpha1.VineyarddSpec{
					Replicas:     0,
					EtcdReplicas: 0,
					Service:      v1alpha1.ServiceConfig{},
					Vineyard:     v1alpha1.VineyardConfig{},
					PluginImage:  v1alpha1.PluginImageConfig{},
					Metric:       v1alpha1.MetricConfig{},
					Volume:       v1alpha1.VolumeConfig{PvcName: "", MountPath: ""},
				},
				Status: v1alpha1.VineyarddStatus{
					ReadyReplicas: 0,
					Conditions:    []appsv1.DeploymentCondition{},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags.Namespace = "vineyard-system"
			flags.VineyarddFile = "/home/zhuyi/v6d/k8s/config/crd/bases/k8s.v6d.io_vineyardds.yaml"
			got, err := BuildVineyardManifestFromFile()
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildVineyardManifestFromFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			a, _ := got.CreationTimestamp.Marshal()
			b, _ := tt.want.CreationTimestamp.Marshal()
			if !reflect.DeepEqual(a, b) {
				t.Errorf("BuildVineyardManifestFromFile() = %v, want %v", got, tt.want)
			}
		})
	}
}

/*func TestNewDeployVineyarddCmd(t *testing.T) {
	tests := []struct {
		name string
		want *cobra.Command
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: deployVineyarddCmd, // 指定预期的 *cobra.Command 值
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewDeployVineyarddCmd(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewDeployVineyarddCmd() = %v, want %v", got, tt.want)
			}
		})
	}
}*/
