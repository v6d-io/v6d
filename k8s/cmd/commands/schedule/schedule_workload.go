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
	"github.com/v6d-io/v6d/k8s/pkg/injector"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

const (
	// VineyardDeploymentSocketPrefix is the prefix of vineyard deployment socket
	VineyardDeploymentSocketPrefix = "/var/run/vineyard-kubernetes/"

	// ScheduleWorkloadMountPath is the mount path of the vineyard socket
	ScheduleWorkloadMountPath = "/var/run"
)

var (
	scheduleWorkloadLong = util.LongDesc(`
	Schedule the workload to a vineyard cluster.
	It will add the podAffinity to the workload so that the workload
	will be scheduled to the vineyard cluster. Besides, if the workload
	does not have the socket volumeMount and volume, it will add one.
	
	Assume you have the following workload yaml:` +
		"\n\n```yaml" + `
	apiVersion: apps/v1
	kind: Deployment
	metadata:
	  name: python-client
	  # Notice, you must set the namespace here
	  namespace: vineyard-job
	spec:
	  selector:
	    matchLabels:
	      app: python
	  template:
	    metadata:
	      labels:
	        app: python
	    spec:
	      containers:
	      - name: python
	        image: python:3.10
	        command: ["python", "-c", "import time; time.sleep(100000)"]` +
		"\n```" + `

	Then you can run the following command to add the podAffinity and socket volume 
	to the workload yaml:

	$ vineyard schedule workload -f workload.yaml -o yaml

	After that, you will get the following workload yaml: ` +
		"\n\n```yaml" + `
	apiVersion: apps/v1
	kind: Deployment
	metadata:
	  creationTimestamp: null
	  name: python-client
	  namespace: vineyard-job
	spec:
	  selector:
	    matchLabels:
	      app: python
	  strategy: {}
	  template:
	   metadata:
	      creationTimestamp: null
	      labels:
	        app: python
	    spec:
	      affinity:
	        podAffinity:
	          requiredDuringSchedulingIgnoredDuringExecution:
	          - labelSelector:
	              matchExpressions:
	              - key: app.kubernetes.io/instance
	                operator: In
	                values:
	                - vineyard-system-vineyardd-sample
	            namespaces:
	            - vineyard-system
	            topologyKey: kubernetes.io/hostname

	      containers:
	      - command:
	        - python
	        - -c
	        - import time; time.sleep(100000)
	        env:
	        - name: VINEYARD_IPC_SOCKET
	          value: /var/run/vineyard.sock
	        image: python:3.10
	        name: python
	        resources: {}
	        volumeMounts:
	        - mountPath: /var/run
	          name: vineyard-socket
	      volumes:
	      - hostPath:
	          path: /var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample
	        name: vineyard-socket` +
		"\n```")

	scheduleWorkloadExample = util.Examples(`
	# Add the podAffinity to the workload yaml
	vineyardctl schedule workload -f workload.yaml \
	--vineyardd-name vineyardd-sample \
	--vineyardd-namespace vineyard-system

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

		client := util.KubernetesClient()
		var obj *unstructured.Unstructured
		if flags.WorkloadFile != "" {
			resource, err := util.ReadFromFile(flags.WorkloadFile)
			if err != nil {
				log.Fatal(err, "failed to read the workload file")
			}
			obj, err = util.ParseManifestToObject(resource)
			if err != nil {
				log.Fatal(err, "failed to parse the workload yaml")
			}

		} else {
			var v bool
			var err error
			obj, v, err = getWorkload(flags.Resource)
			if err != nil || !v {
				log.Fatal(err, "failed to validate the workload")
			}
		}
		obj, err := SchedulingWorkload(client, obj)
		if err != nil {
			log.Fatal(err, "failed to schedule workload")
		}
		if err := output(obj); err != nil {
			log.Fatal(err, "failed to output the workload")
		}
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
	case "Pod":
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
	unstructuredObj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	name := client.ObjectKey{Name: flags.VineyarddName, Namespace: flags.VineyarddNamespace}
	deployment := appsv1.Deployment{}
	if err := c.Get(context.TODO(), name, &deployment); err != nil {
		return nil, errors.Wrap(err, "failed to get the deployment")
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
			"namespaces":  []interface{}{flags.VineyarddNamespace},
		},
	}
	required, err := util.GetRequiredPodAffinity(unstructuredObj)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the required podAffinity")
	}
	required = append(required, requiredDuringSchedulingIgnoredDuringExecution...)

	err = util.SetRequiredPodAffinity(unstructuredObj, required)
	if err != nil {
		return nil, errors.Wrap(err, "failed to set the required podAffinity")
	}

	// check the socket's volumeMount and volume exist or not
	// if not, add the socket's volumeMount and volume to the workload
	socketPath := VineyardDeploymentSocketPrefix + flags.VineyarddNamespace + "/" + flags.VineyarddName
	volumes, err := util.GetVolume(unstructuredObj)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the volumes")
	}
	// if the volume exists, return the workload
	for _, volume := range volumes {
		hostPath := volume.(map[string]interface{})["hostPath"]
		if hostPath != nil {
			path := hostPath.(map[string]interface{})["path"]
			if path != nil && path.(string) == socketPath {
				return unstructuredObj, nil
			}
		}
	}
	// add the socket path volumeMount and volume to the workload
	socketVolume := map[string]interface{}{
		"name": "vineyard-socket",
		"hostPath": map[string]interface{}{
			"path": socketPath,
		},
	}
	volumes = append(volumes, socketVolume)
	err = util.SetVolume(unstructuredObj, volumes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to set the volumes")
	}

	containers, err := util.GetContainer(unstructuredObj)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the containers")
	}
	socketVolumeMount := map[string]interface{}{
		"name":      "vineyard-socket",
		"mountPath": ScheduleWorkloadMountPath,
	}
	for _, container := range containers {
		volumesMounts := container.(map[string]interface{})["volumeMounts"]
		if volumesMounts == nil {
			volumesMounts = []interface{}{}
		}
		volumesMounts = append(volumesMounts.([]interface{}), socketVolumeMount)
		container.(map[string]interface{})["volumeMounts"] = volumesMounts
	}

	injector.InjectEnv(&containers, ScheduleWorkloadMountPath)

	err = util.SetContainer(unstructuredObj, containers)
	if err != nil {
		return nil, errors.Wrap(err, "failed to set the containers")
	}

	return unstructuredObj, nil
}

func output(unstructuredObj *unstructured.Unstructured) error {
	ss, err := unstructuredObj.MarshalJSON()
	if err != nil {
		return errors.Wrap(err, "failed to marshal the unstructuredObj")
	}
	jsonStr := string(ss)
	if flags.ScheduleOutputFormat == "json" {
		log.Output(jsonStr)
	} else {
		yamlStr, err := util.ConvertToYaml(jsonStr)
		if err != nil {
			return errors.Wrap(err, "failed to convert to yaml")
		}
		log.Output(yamlStr)
	}
	return nil
}
