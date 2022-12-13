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
	"strconv"

	vineyardV1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/schedulers"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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
		obj, _, err := decode(resources[i], nil, nil)
		if err != nil {
			fmt.Println("failed to decode resource", err)
			os.Exit(1)
		}

		proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		if err != nil {
			fmt.Println("failed to convert resource", err)
			os.Exit(1)
		}

		unstructuredObj := &unstructured.Unstructured{Object: proto}
		if success := SchedulingWorkload(c, unstructuredObj, kubeconfig); !success {
			fmt.Println("failed to schedule resource", err)
			os.Exit(1)
		}
	}
}

func getWaitCondition(kind string) (string, string) {
	switch kind {
	case "Deployment":
		return "available", "deployment.apps"
	case "StatefulSet":
		return "available", "statefulset.apps"
	case "DaemonSet":
		return "available", "daemonset.apps"
	case "ReplicaSet":
		return "available", "replicaset.apps"
	case "Job":
		return "complete", "job.batch"
	case "CronJob":
		return "complete", "cronjob.batch"
	default:
		return "", ""
	}
}

// wait for the workload to be ready
func waitWorkload(c client.Client, kind, kubeconfig, condition, apiVersion, name, namespace string) error {
	command := "kubectl wait --for=condition=" + condition + " --timeout=60s " + apiVersion +
		"/" + name + " -n " + namespace + " --kubeconfig=" + kubeconfig
	// run kubectl wait command
	cmd := exec.Command("bash", "-c", command)

	err := cmd.Run()
	if err != nil {
		fmt.Println("failed to run kubectl wait, please check", err)
		return err
	}
	return nil
}

// SchedulingWorkload is used to schedule the workload
func SchedulingWorkload(c client.Client, obj *unstructured.Unstructured, kubeconfig string) bool {
	// get template labels
	l, _, err := unstructured.NestedStringMap(obj.Object, "spec", "template", "metadata", "labels")
	if err != nil {
		fmt.Println("failed to get labels", err)
		return false
	}

	// get template annotations
	a, _, err := unstructured.NestedStringMap(obj.Object, "spec", "template", "metadata", "annotations")
	if err != nil {
		fmt.Println("failed to get annotations", err)
		return false
	}

	// get name and namespace
	name, _, err := unstructured.NestedString(obj.Object, "metadata", "name")
	if err != nil {
		fmt.Println("failed to get the name of resource", err)
		return false
	}
	namespace, _, err := unstructured.NestedString(obj.Object, "metadata", "namespace")
	if err != nil {
		fmt.Println("failed to get the namespace of resource", err)
		return false
	}

	// get obj kind
	kind, _, err := unstructured.NestedString(obj.Object, "kind")
	if err != nil {
		fmt.Println("failed to get the kind of resource", err)
		return false
	}

	condition, apiVersion := getWaitCondition(kind)

	// for non-workload resources
	if condition == "" {
		if err := c.Create(context.TODO(), obj); err != nil {
			fmt.Println("failed to create non-workload resources", err)
			return false
		}
		return true
	}

	// for workload resources
	r := l[labels.WorkloadReplicas]
	replicas, err := strconv.Atoi(r)
	if err != nil {
		fmt.Println("failed to get replicas", err)
		return false
	}

	str, required, localObjects, globalObjects := Scheduling(c, a, l, replicas, namespace)

	// setup annotations and labels
	l["scheduling.v6d.io/enabled"] = "true"
	if err := unstructured.SetNestedStringMap(obj.Object, l, "spec", "template", "metadata", "labels"); err != nil {
		fmt.Println("failed to set labels", err)
		return false
	}

	a["scheduledOrder"] = str
	if err := unstructured.SetNestedStringMap(obj.Object, a, "spec", "template", "metadata", "annotations"); err != nil {
		fmt.Println("failed to set annotations", err)
		return false
	}

	if err := c.Create(context.TODO(), obj); err != nil {
		fmt.Println("failed to create workload", err)
		return false
	}

	// wait for the workload to be ready
	if err := waitWorkload(c, kind, kubeconfig, condition, apiVersion, name, namespace); err != nil {
		fmt.Println("failed to wait workload", err)
		return false
	}

	// get the workload
	if err := c.Get(context.TODO(), types.NamespacedName{Name: name, Namespace: namespace}, obj); err != nil {
		fmt.Println("failed to get workload", err)
		return false
	}
	version, _, err := unstructured.NestedString(obj.Object, "apiVersion")
	if err != nil {
		fmt.Println("failed to get apiVersion", err)
		return false
	}

	uid, _, err := unstructured.NestedString(obj.Object, "metadata", "uid")
	if err != nil {
		fmt.Println("failed to get uid", err)
		return false
	}
	ownerReferences := []metav1.OwnerReference{
		{
			APIVersion: version,
			Kind:       kind,
			Name:       name,
			UID:        types.UID(uid),
		},
	}

	if err := schedulers.CreateConfigmapForID(c, slog, required, namespace, localObjects, globalObjects, ownerReferences); err != nil {
		slog.Info(fmt.Sprintf("can't create configmap for object ID %v", err))
	}

	return true
}
