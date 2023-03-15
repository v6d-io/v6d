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

package schedule

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/schedulers"
)

var (
	scheduleWorkflowLong = util.LongDesc(`
	Schedule a workflow based on the vineyard cluster.
	It will apply the workflow to kubernetes cluster and deploy the workload
	of the workflow on the vineyard cluster with the best-fit strategy.`)

	scheduleWorkflowExample = util.Examples(`
	# schedule a workflow to the vineyard cluster with the best-fit strategy
	vineyardctl schedule workflow --file workflow.yaml`)

	ownerReferences []metav1.OwnerReference

	jobKind = "Job"
)

// scheduleWorkflowCmd schedules a workflow based on the vineyard cluster
var scheduleWorkflowCmd = &cobra.Command{
	Use:     "workflow",
	Short:   "Schedule a workflow based on the vineyard cluster",
	Long:    scheduleWorkflowLong,
	Example: scheduleWorkflowExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		client := util.KubernetesClient()
		manifests, err := util.ReadFromFile(flags.WorkflowFile)
		if err != nil {
			log.Fatal(err, "failed to read workflow file")
		}

		objs, err := util.ParseManifestsToObjects([]byte(manifests))
		if err != nil {
			log.Fatal(err, "failed to parse workflow file")
		}

		for _, obj := range objs {
			if err := SchedulingWorkflow(client, obj); err != nil {
				log.Fatal(err, "failed to schedule workload")
			}
		}

		log.Info("Scheduling workflow successfully.")
	},
}

func NewScheduleWorkflowCmd() *cobra.Command {
	return scheduleWorkflowCmd
}

func init() {
	flags.ApplySchedulerWorkflowOpts(scheduleWorkflowCmd)
}

// SchedulingWorkflow is used to schedule the workload of the workflow
func SchedulingWorkflow(c client.Client, obj *unstructured.Unstructured) error {
	// get template labels
	l, _, err := unstructured.NestedStringMap(obj.Object, "spec", "template", "metadata", "labels")
	if err != nil {
		return errors.Wrap(err, "failed to get labels")
	}

	// get template annotations
	a, _, err := unstructured.NestedStringMap(
		obj.Object,
		"spec",
		"template",
		"metadata",
		"annotations",
	)
	if err != nil {
		return errors.Wrap(err, "failed to get annotations")
	}

	// get name and namespace
	name, _, err := unstructured.NestedString(obj.Object, "metadata", "name")
	if err != nil {
		return errors.Wrap(err, "failed to get the name of resource")
	}
	namespace, _, err := unstructured.NestedString(obj.Object, "metadata", "namespace")
	if err != nil {
		return errors.Wrap(err, "failed to get the namespace of resource")
	}

	// get obj kind
	kind, _, err := unstructured.NestedString(obj.Object, "kind")
	if err != nil {
		return errors.Wrap(err, "failed to get the kind of resource")
	}

	isWorkload := ValidateWorkloadKind(kind)

	// for non-workload resources
	if !isWorkload {
		if err := c.Create(context.TODO(), obj); err != nil {
			return errors.Wrap(err, "failed to create non-workload resources")
		}
		return nil // skip scheduling for non-workload resources
	}

	// for workload resources
	r := l[labels.WorkloadReplicas]
	replicas, err := strconv.Atoi(r)
	if err != nil {
		return errors.Wrap(err, "failed to get replicas")
	}

	str := Scheduling(c, a, l, replicas, namespace, ownerReferences)

	// setup annotations and labels
	l[labels.SchedulingEnabledLabel] = "true"
	if err := unstructured.SetNestedStringMap(obj.Object, l, "spec", "template", "metadata", "labels"); err != nil {
		return errors.Wrap(err, "failed to set labels")
	}

	a["scheduledOrder"] = str
	if err := unstructured.SetNestedStringMap(obj.Object, a, "spec", "template", "metadata", "annotations"); err != nil {
		return errors.Wrap(err, "failed to set annotations")
	}

	var parallelism int64
	// for the Job of Kubernetes resources, we need to get the parallelism
	if kind == jobKind {
		parallelism, _, err = unstructured.NestedInt64(obj.Object, "spec", "parallelism")
		if err != nil {
			return errors.Wrap(err, "failed to get completions of job")
		}
	}
	if err := util.Create(c, obj, func(obj *unstructured.Unstructured) bool {
		status, _, err := unstructured.NestedMap(obj.Object, "status")
		if err != nil {
			return false
		}
		switch kind {
		case "Deployment":
			return status["availableReplicas"] == status["replicas"]
		case "StatefulSet":
			return status["readyReplicas"] == status["replicas"]
		case "DaemonSet":
			return status["numberReady"] == status["desiredNumberScheduled"]
		case "ReplicaSet":
			return status["availableReplicas"] == status["replicas"]
		case "Job":
			if status["succeeded"] == nil {
				return false
			}
			return status["succeeded"].(int64) == parallelism
		case "CronJob":
			return status["lastScheduleTime"] != ""
		}
		return true
	}); err != nil {
		log.Fatalf(err, "failed to wait the workload %s/%s", namespace, name)
	}

	// use the previous workload as ownerReference
	if err := c.Get(context.TODO(), types.NamespacedName{Name: name, Namespace: namespace}, obj); err != nil {
		return errors.Wrap(err, "failed to get workload")
	}
	version, _, err := unstructured.NestedString(obj.Object, "apiVersion")
	if err != nil {
		return errors.Wrap(err, "failed to get apiVersion")
	}

	uid, _, err := unstructured.NestedString(obj.Object, "metadata", "uid")
	if err != nil {
		return errors.Wrap(err, "failed to get uid")
	}
	ownerReferences = []metav1.OwnerReference{
		{
			APIVersion: version,
			Kind:       kind,
			Name:       name,
			UID:        types.UID(uid),
		},
	}
	return nil
}

// Scheduling is used to schedule jobs to nodes
func Scheduling(c client.Client, a, l map[string]string, replica int, namespace string,
	ownerReferences []metav1.OwnerReference,
) string {
	scheduledOrder := ""
	jobToNode := make(map[string]int)
	// get all nodes that have vineyardd
	vineyarddName := l[labels.VineyarddName]
	vineyarddNamespace := l[labels.VineyarddNamespace]
	allNodes := schedulers.GetVineyarddNodes(c, log.Log, vineyarddName, vineyarddNamespace)
	// use round-robin to schedule workload here
	if a["scheduling.k8s.v6d.io/required"] == "none" {
		l := len(allNodes)
		for i := 0; i < replica; i++ {
			jobToNode[allNodes[i%l]]++
		}

		s := make([]string, 0)
		for n, v := range jobToNode {
			s = append(s, n+"="+strconv.Itoa(v))
		}
		scheduledOrder = strings.Join(s, ",")
		return scheduledOrder
	}

	// get required jobs
	required, err := schedulers.GetRequiredJob(log.Log, a)
	if err != nil {
		log.Info(fmt.Sprintf("get required jobs failed: %v", err))
		return ""
	}
	// get all global objects
	globalObjects, err := schedulers.GetGlobalObjectsByID(c, log.Log, required)
	if err != nil {
		log.Info(fmt.Sprintf("get global objects failed: %v", err))
		return ""
	}

	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := schedulers.GetLocalObjectsBySignatures(c, log.Log, localsigs)
	if err != nil {
		log.Info(fmt.Sprintf("get local objects failed: %v", err))
		return ""
	}

	locations, nchunks, nodes := schedulers.GetObjectInfo(localObjects, int64(replica))

	var cnt int64

	for i := 0; i < replica; i++ {
		rank := int64(i)
		for _, node := range nodes {
			localfrags := int64(len(locations[node]))
			if cnt+localfrags >= (nchunks*rank + (nchunks+1)/2) {
				jobToNode[node]++
				break
			}
			cnt += localfrags
		}
	}

	s := make([]string, 0)
	for n, v := range jobToNode {
		s = append(s, n+"="+strconv.Itoa(v))
	}
	scheduledOrder = strings.Join(s, ",")

	if err := schedulers.CreateConfigmapForID(c, log.Log, required, namespace, localObjects, globalObjects, ownerReferences); err != nil {
		log.Info(fmt.Sprintf("can't create configmap for object ID %v", err))
	}

	return scheduledOrder
}
