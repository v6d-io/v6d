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
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"go.uber.org/multierr"
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
	vineyardctl schedule workflow --file workflow.yaml
	
	# schedule a workflow without CRD installed
	# Notice, it only works for the workflow built by pods
	vineyardctl schedule workflow --file pod-workflow.yaml --without-crd`)

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
	var errList error

	l, err := util.GetLabels(obj)
	_ = multierr.Append(errList, err)

	a, err := util.GetAnnotations(obj)
	_ = multierr.Append(errList, err)

	name, err := util.GetName(obj)
	_ = multierr.Append(errList, err)

	namespace, err := util.GetNamespace(obj)
	_ = multierr.Append(errList, err)

	// get obj kind
	kind := obj.GetKind()
	_ = multierr.Append(errList, err)

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

	scheduler := schedulers.NewVineyardSchedulerOutsideCluster(
		c, a, l, namespace, ownerReferences)

	err = scheduler.SetupConfig()
	if err != nil {
		return errors.Wrap(err, "failed to setup scheduler config")
	}

	result, err := scheduler.SetWithoutCRD(flags.WithoutCRD).Schedule(replicas)
	if err != nil {
		return errors.Wrap(err, "failed to schedule workload")
	}

	// setup annotations and labels
	l[labels.SchedulingEnabledLabel] = "true"
	if err := util.SetLabels(obj, l); err != nil {
		return errors.Wrap(err, "failed to set labels")
	}

	if a == nil {
		a = make(map[string]string)
	}

	a["scheduledOrder"] = result
	if err := util.SetAnnotations(obj, a); err != nil {
		return errors.Wrap(err, "failed to set annotations")
	}

	// set the nodename for the pod directly if there is no CRD installed
	if flags.WithoutCRD {
		node := strings.Split(result, "=")[0]
		if err := util.SetNodename(obj, node); err != nil {
			return errors.Wrap(err, "failed to set node name")
		}
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
		status, err := util.GetStatus(obj)
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
	if err := c.Get(context.TODO(), types.NamespacedName{Name: name,
		Namespace: namespace}, obj); err != nil {
		return errors.Wrap(err, "failed to get workload")
	}

	ownerReferences = []metav1.OwnerReference{
		{
			APIVersion: obj.GetAPIVersion(),
			Kind:       kind,
			Name:       name,
			UID:        obj.GetUID(),
		},
	}
	return nil
}
