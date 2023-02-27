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

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

// scheduleWorkloadCmd schedules the workload to a vineyard cluster
var scheduleWorkloadCmd = &cobra.Command{
	Use:   "workload",
	Short: "scheduleWorkload schedules the workload to a vineyard cluster",
	Long: `scheduleWorkload schedules the workload to a vineyard cluster. It will
add the podAffinity to the workload so that the workload will be scheduled to the
vineyard cluster. For example:

# Add the podAffinity to the workload for the specific vineyard cluster
vineyardctl dryschedule workload --resource '{
	"apiVersion": "apps/v1",
	"kind": "Deployment",
	"metadata": {
	  "name": "web-server"
	},
	"spec": {
	  "selector": {
		"matchLabels": {
		  "app": "web-store"
		}
	  },
	  "replicas": 3,
	  "template": {
		"metadata": {
		  "labels": {
			"app": "web-store"
		  }
		},
		"spec": {
		  "affinity": {
			"podAntiAffinity": {
			  "requiredDuringSchedulingIgnoredDuringExecution": [
				{
				  "labelSelector": {
					"matchExpressions": [
					  {
						"key": "app",
						"operator": "In",
						"values": [
						  "web-store"
						]
					  }
					]
				  },
				  "topologyKey": "kubernetes.io/hostname"
				}
			  ]
			},
			"podAffinity": {
			  "requiredDuringSchedulingIgnoredDuringExecution": [
				{
				  "labelSelector": {
					"matchExpressions": [
					  {
						"key": "app",
						"operator": "In",
						"values": [
						  "store"
						]
					  }
					]
				  },
				  "topologyKey": "kubernetes.io/hostname"
				}
			  ]
			}
		  },
		  "containers": [
			{
			  "name": "web-app",
			  "image": "nginx:1.16-alpine"
			}
		  ]
		}
	  }
	}
  }' --vineyardd-name vineyardd-sample --vineyardd-namespace vineyard-system`,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		if err := validateWorkload(flags.Resource); err != nil {
			util.ErrLogger.Fatal("failed to validate the workload: ", err)
		}
		client := util.KubernetesClient()

		workload, err := SchedulingWorkload(client)
		if err != nil {
			util.ErrLogger.Fatal("failed to schedule workload: ", err)
		}

		util.InfoLogger.Println(workload)
	},
}

func NewScheduleWorkloadCmd() *cobra.Command {
	return scheduleWorkloadCmd
}

func init() {
	flags.ApplySchedulerOpts(scheduleWorkloadCmd)
}

func validateWorkload(workload string) error {
	decoder := util.Deserializer()
	obj, _, err := decoder.Decode([]byte(workload), nil, nil)
	if err != nil {
		return errors.Wrap(err, "failed to decode the workload")
	}
	kind := obj.GetObjectKind().GroupVersionKind().Kind
	switch kind {
	case "Deployment":
		return nil
	case "StatefulSet":
		return nil
	case "ReplicaSet":
		return nil
	case "Job":
		return nil
	case "CronJob":
		return nil
	}
	return errors.Wrap(err, "the workload kind is not supported")
}

// SchedulingWorkload is used to schedule the workload to the vineyard cluster
// and add the podAffinity to the workload
func SchedulingWorkload(c client.Client) (string, error) {
	resource := flags.Resource
	name := client.ObjectKey{Name: flags.VineyarddName, Namespace: flags.VineyarddNamespace}
	deployment := appsv1.Deployment{}
	if err := c.Get(context.TODO(), name, &deployment); err != nil {
		return "", errors.Wrap(err, "failed to get the deployment")
	}
	newPodAffinity := corev1.PodAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "app.kubernetes.io/instance",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"vineyardd"},
						},
					},
				},
				TopologyKey: "kubernetes.io/hostname",
			},
		},
	}

	decoder := util.Deserializer()
	obj, _, err := decoder.Decode([]byte(resource), nil, nil)
	if err != nil {
		return "", errors.Wrap(err, "failed to decode the workload")
	}

	proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return "", errors.Wrap(err, "failed to convert the workload to unstructured")
	}

	unstructuredObj := &unstructured.Unstructured{Object: proto}

	spec := unstructuredObj.Object["spec"].(map[string]interface{})["template"].(map[string]interface{})["spec"].(map[string]interface{})
	if spec["affinity"] == nil {
		spec["affinity"] = make(map[string]interface{})
	}

	affinity := spec["affinity"].(map[string]interface{})
	if affinity["podAffinity"] == nil {
		affinity["podAffinity"] = newPodAffinity
	} else {
		podAffinity := affinity["podAffinity"].(map[string]interface{})
		if podAffinity["requiredDuringSchedulingIgnoredDuringExecution"] == nil {
			podAffinity["requiredDuringSchedulingIgnoredDuringExecution"] = newPodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
		} else {
			required := podAffinity["requiredDuringSchedulingIgnoredDuringExecution"].([]interface{})
			required = append(required, newPodAffinity.RequiredDuringSchedulingIgnoredDuringExecution)
			podAffinity["requiredDuringSchedulingIgnoredDuringExecution"] = required
		}
	}

	ss, err := unstructuredObj.MarshalJSON()
	if err != nil {
		return "", errors.Wrap(err, "failed to marshal the unstructuredObj")
	}
	return string(ss), nil
}