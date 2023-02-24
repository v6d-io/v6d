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

package dryschedule

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoScheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// dryScheduleWorkload schedules the workload to a vineyard cluster
var dryScheduleWorkloadCmd = &cobra.Command{
	Use:   "workload",
	Short: "DryScheduleWorkload schedules the workload to a vineyard cluster",
	Long: `DryScheduleWorkload schedules the workload to a vineyard cluster. It will
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
		if err := util.ValidateNoArgs("dryScheduleWorkloadCmd workload", args); err != nil {
			util.ErrLogger.Fatal("failed to validate dryScheduleWorkloadCmd workload command args and flags: ", err,
				"the extra args are: ", args)
		}

		if err := validateWorkload(flags.Resource); err != nil {
			util.ErrLogger.Fatal("failed to validate the workload: ", err)
		}

		scheme, err := util.GetClientgoScheme()
		if err != nil {
			util.ErrLogger.Fatal("failed to get client-go scheme: ", err)
		}

		kubeclient, err := util.GetKubeClient(scheme)
		if err != nil {
			util.ErrLogger.Fatal("failed to get kube client: ", err)
		}

		workload, err := SchedulingWorkload(kubeclient)
		if err != nil {
			util.ErrLogger.Fatal("failed to schedule workload: ", err)
		}

		fmt.Println(workload)
	},
}

func NewScheduleWorkloadCmd() *cobra.Command {
	return dryScheduleWorkloadCmd
}

func init() {
	flags.NewDrySchedulerOpts(dryScheduleWorkloadCmd)
}

func validateWorkload(workload string) error {
	decode := clientgoScheme.Codecs.UniversalDeserializer().Decode
	obj, _, err := decode([]byte(workload), nil, nil)
	if err != nil {
		return fmt.Errorf("failed to decode the workload: %w", err)
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
	return fmt.Errorf("the workload kind is not supported")
}

// SchedulingWorkload is used to schedule the workload to the vineyard cluster
// and add the podAffinity to the workload
func SchedulingWorkload(c client.Client) (string, error) {
	resource := flags.Resource
	name := client.ObjectKey{Name: flags.VineyarddName, Namespace: flags.VineyarddNamespace}
	deployment := appsv1.Deployment{}
	if err := c.Get(context.TODO(), name, &deployment); err != nil {
		return "", fmt.Errorf("failed to get the deployment: ", err)
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

	decode := clientgoScheme.Codecs.UniversalDeserializer().Decode
	obj, _, err := decode([]byte(resource), nil, nil)
	if err != nil {
		return "", fmt.Errorf("failed to decode the workload: %w", err)
	}

	proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return "", fmt.Errorf("failed to convert the workload to unstructured: %w", err)
	}

	unstructuredObj := &unstructured.Unstructured{Object: proto}

	spec := unstructuredObj.Object["spec"].(map[string]interface{})["template"].(map[string]interface{})["spec"].(map[string]interface{})
	if spec["affinity"] == nil {
		spec["affinity"] = make(map[string]interface{})
	}

	affinity := unstructuredObj.Object["spec"].(map[string]interface{})["template"].(map[string]interface{})["spec"].(map[string]interface{})["affinity"].(map[string]interface{})
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
		return "", fmt.Errorf("failed to marshal the unstructuredObj: %w", err)
	}
	return string(ss), nil
}
