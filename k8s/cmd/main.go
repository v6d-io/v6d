/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

/* Pakcage cmd is used for simplify the operator usage. */
package main

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	vineyardV1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientgoScheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var (
	scheme = runtime.NewScheme()
)

// add k8s apis scheme and apiextensions scheme
func init() {
	_ = clientgoScheme.AddToScheme(scheme)

	_ = vineyardV1alpha1.AddToScheme(scheme)
}

func main() {
	args := os.Args
	if len(args) < 3 {
		fmt.Println("Usage: go run main.go scheduler.go [kubeconfig path] [workflow path]")
		os.Exit(1)
	}
	kubeconfig := args[1]
	workflow := args[2]

	cfg, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		panic(err)
	}

	client, _ := client.New(cfg, client.Options{Scheme: scheme})
	//read file from path
	contents, err := ioutil.ReadFile(workflow)
	if err != nil {
		fmt.Println("Error reading file:", err)
	}

	// parse the workflow yaml file
	parseManifests(client, contents, kubeconfig)
}

func parseManifests(c client.Client, manifests []byte, kubeconfig string) {
	// parse the workflow yaml file
	resources := bytes.Split(manifests, []byte("---"))
	for i := range resources {
		// parse each resource
		if resources[i][0] == '\r' {
			resources[i] = resources[i][1:]
		}
		decode := clientgoScheme.Codecs.UniversalDeserializer().Decode
		obj, gvk, err := decode(resources[i], nil, nil)
		if err != nil {
			fmt.Println("failed to decode resource", err)
			os.Exit(1)
		}
		if success := SchedulingWorkload(c, gvk, obj, kubeconfig); !success {
			fmt.Println("failed to wait resource", err)
			os.Exit(1)
		}
	}
}

func scheduling(c client.Client, manifestsQueue []interface{}) {
	// scheduling
	vineyarddList := v1alpha1.VineyarddList{}
	//podList := v1.PodList{}
	if err := c.List(context.TODO(), &vineyarddList); err != nil {
		fmt.Println("failed to list vineyardd", err)
		os.Exit(1)
	}

	deploymentList := appsv1.DeploymentList{}
	if err := c.List(context.TODO(), &deploymentList); err != nil {
		fmt.Println("failed to list deployment", err)
		os.Exit(1)
	}
	fmt.Println("vineyarddList: ", vineyarddList)
	for _, q := range manifestsQueue {
		deployment := q.(*appsv1.Deployment)
		anno := deployment.Spec.Template.Annotations
		if anno["scheduling.k8s.v6d.io/required"] == "none" {
			if err := c.Create(context.TODO(), deployment); err != nil {
				fmt.Println("failed to create deployment", err)
				os.Exit(1)
			}
		}

		fmt.Println("deployment: ", deployment)

	}
}

// SchedulingWorkload is used to schedule the workload
func SchedulingWorkload(c client.Client, gvk *schema.GroupVersionKind, obj interface{}, kubeconfig string) bool {
	command := ""
	switch gvk.Kind {
	case "Deployment":
		deployment := obj.(*appsv1.Deployment)
		annotations := deployment.Spec.Template.Annotations
		labels := deployment.Spec.Template.Labels
		str := Scheduling(c, annotations, labels, int(*deployment.Spec.Replicas), deployment.Namespace)
		annotations["scheduledOrder"] = str
		labels["scheduling.v6d.io/enabled"] = "true"
		if err := c.Create(context.TODO(), deployment); err != nil {
			fmt.Println("failed to create deployment", err)
			os.Exit(1)
		}
		command = "kubectl wait --for=condition=available --timeout=60s deployment.apps/" + deployment.Name +
			" -n " + deployment.Namespace + " --kubeconfig=" + kubeconfig
	case "StatefulSet":
		statefulset := obj.(*appsv1.StatefulSet)
		command = "kubectl wait --for=condition=available --timeout=60s statefulset.apps/" + statefulset.Name +
			" -n " + statefulset.Namespace + " --kubeconfig=" + kubeconfig
	case "Job":
		job := obj.(*batchv1.Job)
		annotations := job.Spec.Template.Annotations
		labels := job.Spec.Template.Labels
		str := Scheduling(c, annotations, labels, int(*job.Spec.Parallelism), job.Namespace)
		annotations["scheduledOrder"] = str
		labels["scheduling.v6d.io/enabled"] = "true"
		if err := c.Create(context.TODO(), job); err != nil {
			fmt.Println("failed to create job", err)
			os.Exit(1)
		}
		command = "kubectl wait --for=condition=complete --timeout=60s job.batch/" + job.Name +
			" -n " + job.Namespace + " --kubeconfig=" + kubeconfig
	case "DaemonSet":
		daemonset := obj.(*appsv1.DaemonSet)
		command = "kubectl wait --for=condition=available --timeout=60s daemonset.apps/" + daemonset.Name +
			" -n " + daemonset.Namespace + " --kubeconfig=" + kubeconfig
	case "ReplicaSet":
		replicaset := obj.(*appsv1.ReplicaSet)
		command = "kubectl wait --for=condition=available --timeout=60s replicaset.apps/" + replicaset.Name +
			" -n " + replicaset.Namespace + " --kubeconfig=" + kubeconfig
	case "CronJob":
		cronjob := obj.(*batchv1.CronJob)
		command = "kubectl wait --for=condition=complete --timeout=60s cronjob.batch/" + cronjob.Name +
			" -n " + cronjob.Namespace + " --kubeconfig=" + kubeconfig
	default:
		fmt.Println("unsupported workload kind: ", gvk.Kind)
		os.Exit(1)
		return false
	}

	// run kubectl wait command
	cmd := exec.Command("bash", "-c", command)

	err := cmd.Run()
	if err != nil {
		fmt.Println("failed to run kubectl wait, please check", err)
		os.Exit(1)
	}
	return true
}
