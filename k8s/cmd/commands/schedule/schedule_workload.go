/** Copyright 2020-2023 Alibaba Group Holding Limited.

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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	scheduleWorkloadLong = util.LongDesc(`
	Schedule the workload to a vineyard cluster.
	It will add the podAffinity to the workload so that the workload
	will be scheduled to the vineyard cluster.`)

	scheduleWorkloadExample = util.Examples(`
	# Add the podAffinity to the workload for the specific vineyard cluster
	vineyardctl schedule workload --resource '{
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
		}' \
		--vineyardd-name vineyardd-sample \
		--vineyardd-namespace vineyard-system`)
)

// scheduleWorkloadCmd schedules the workload to a vineyard cluster
var scheduleWorkloadCmd = &cobra.Command{
	Use:     "workload",
	Short:   "Schedule the workload to a vineyard cluster",
	Long:    scheduleWorkloadLong,
	Example: scheduleWorkloadExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		obj, v, err := getWorkload(flags.Resource)
		if err != nil || !v {
			log.Fatal(err, "failed to validate the workload")
		}
		client := util.KubernetesClient()

		workload, err := SchedulingWorkload(client, obj)
		if err != nil {
			log.Fatal(err, "failed to schedule workload")
		}

		log.Output(workload)
	},
}

func NewScheduleWorkloadCmd() *cobra.Command {
	return scheduleWorkloadCmd
}

func init() {
	flags.ApplySchedulerWorkloadOpts(scheduleWorkloadCmd)
}

func ValidateWorkloadKind(kind string) bool {
	isWorkload := true
	switch kind {
	case "Deployment":
		return isWorkload
	case "StatefulSet":
		return isWorkload
	case "ReplicaSet":
		return isWorkload
	case "Job":
		return isWorkload
	case "CronJob":
		return isWorkload
	}
	return !isWorkload
}

func getWorkload(workload string) (*unstructured.Unstructured, bool, error) {
	isWorkload := true
	obj, err := util.ParseManifestToObject(workload)
	if err != nil {
		return obj, isWorkload, errors.Wrap(err, "failed to parse the workload")
	}
	kind := obj.GetObjectKind().GroupVersionKind().Kind
	return obj, ValidateWorkloadKind(kind), nil
}

// SchedulingWorkload is used to schedule the workload to the vineyard cluster
// and add the podAffinity to the workload
func SchedulingWorkload(c client.Client,
	unstructuredObj *unstructured.Unstructured) (string, error) {
	name := client.ObjectKey{Name: flags.VineyarddName, Namespace: flags.VineyarddNamespace}
	deployment := appsv1.Deployment{}
	if err := c.Get(context.TODO(), name, &deployment); err != nil {
		return "", errors.Wrap(err, "failed to get the deployment")
	}
	value := flags.VineyarddNamespace + "-" + flags.VineyarddName

	requiredDuringSchedulingIgnoredDuringExecution := []interface{}{
		map[string]interface{}{
			"labelSelector": map[string]interface{}{
				"matchExpressions": []interface{}{
					map[string]interface{}{
						"key":      "app.kubernetes.io/instance",
						"operator": "In",
						"values":   []interface{}{value},
					},
				},
			},
			"topologyKey": "kubernetes.io/hostname",
		},
	}

	var required []interface{}
	required, _, _ = unstructured.NestedSlice(unstructuredObj.Object,
		"spec", "template", "spec", "affinity", "podAffinity", "requiredDuringSchedulingIgnoredDuringExecution")
	required = append(required, requiredDuringSchedulingIgnoredDuringExecution...)

	err := unstructured.SetNestedSlice(unstructuredObj.Object, required,
		"spec", "template", "spec", "affinity", "podAffinity", "requiredDuringSchedulingIgnoredDuringExecution")
	if err != nil {
		return "", errors.Wrap(err, "failed to set the nested slice")
	}

	ss, err := unstructuredObj.MarshalJSON()
	if err != nil {
		return "", errors.Wrap(err, "failed to marshal the unstructuredObj")
	}
	return string(ss), nil
}
