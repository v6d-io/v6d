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
package inject

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	injectArgoWorkflowLong = util.LongDesc(`
	Inject the vineyard volumes into the argo workflow DAGs. You can input
	the workflow manifest file and the injected manifest file with 
	vineyard volume will be output to the file with the suffix 
	"_with_vineyard", such as "workflow_with_vineyard.yaml".

	Suppose the workflow manifest named "workflow.yaml" is as follows:` +
		"\n\n```yaml" + `
	apiVersion: argoproj.io/v1alpha1
	kind: Workflow
	metadata:
	  generateName: mlops-
	spec:
	  entrypoint: dag
	  templates:
	  - name: producer
	    container:
	      image: producer:latest
	      command: [python]
	      args: ["/producer.py"]
	  - name: consumer
	    container:
	      image: consumer:latest
	      command: [python]
	      args: ["/consumer.py"]
	  - name: dag
	    dag:
	      tasks:
	      - name: producer
	        template: producer
	      - name: consumer
	        template: consumer
	        dependencies:
	          - producer` +
		"\n```" + `
	
	Assume the 'producer' and 'consumer' task all need to use vineyard 
	volume, you can inject the vineyard volume into the workflow manifest 
	with the following command:
	
	$ vineyardctl inject argo-workflow -f workflow.yaml \
	      --templates="producer,consumer" \
		  --vineyard-cluster="vineyard-system/vineyardd-sample" \
		  --mount-path="/vineyard/data" \
		  --dag="dag" \
		  --tasks="producer,consumer" \
		  --output-as-file

	The injected manifest will be output to the file named "workflow_with_vineyard.yaml".
	
	$ cat workflow_with_vineyard.yaml` +
		"\n\n```yaml" + `
	apiVersion: argoproj.io/v1alpha1
	kind: Workflow
	metadata:
	  generateName: mlops-
	spec:
	  entrypoint: dag
	  templates:
	  - name: producer
	    container:
	      image: producer:latest
	      command: [python]
	      args: ["/producer.py"]
	      ################## Injected #################
	      volumeMounts: 
	      - name: vineyard-objects
	        mountPath: /vineyard/data
	      #############################################
	    ######################## Injected #######################
	    volumes:
	    - name: vineyard-objects
	      persistentVolumeClaim: 
	        claimName: '{{inputs.parameters.vineyard-objects-name}}'
	    #########################################################
	    ############## Injected #############
	    inputs:
	      parameters:
	      - {name: vineyard-objects-name}
	    #####################################
	  - name: consumer
	    container:
	      image: consumer:latest
	      command: [python]
	      args: ["/consumer.py"]
	      ################## Injected #################
	      volumeMounts: 
	      - name: vineyard-objects
	        mountPath: /vineyard/data
	      #############################################
	    ######################## Injected #######################
	    volumes:
	    - name: vineyard-objects
	      persistentVolumeClaim: 
	        claimName: '{{inputs.parameters.vineyard-objects-name}}'
	    #########################################################
	    ############## Injected #############
	    inputs:
	      parameters:
	      - {name: vineyard-objects-name}
	    #####################################
	  - name: dag
	    dag:
	      tasks:
	      - name: producer
	        template: producer
	        arguments:
	          parameters:
	          ################################# Injected ################################
	          - name: vineyard-objects-name
	            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
	          ###########################################################################
	        dependencies:
	          ########### Injected ##########
	          - vineyard-objects
	          ###############################
	      - name: consumer
	        template: consumer
	        arguments:
	          parameters:
	          ################################# Injected ################################
	          - name: vineyard-objects-name
	            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
	          ###########################################################################
	        dependencies:
	          - producer
	          ########### Injected ##########
	          - vineyard-objects
	          ###############################
	      ########## Injected #########
	      - name: vineyard-objects
	        template: vineyard-objects
	      #############################
	############################# Injected ##########################
	  - name: vineyard-objects
	    resource:
	      action: create
	      setOwnerReference: true
	      manifest: |
	        apiVersion: v1
	        kind: PersistentVolumeClaim
	        metadata:
	          name: '{{workflow.name}}-vineyard-objects-pvc'
	        spec:
	          accessModes:
	          - ReadWriteMany
	          resources:
	            requests:
	              storage: 1Mi
	          storageClassName: vineyard-system.vineyardd-sample.csi
	    outputs:
	      parameters:
	      - name: vineyard-objects-name
	        valueFrom:
	          jsonPath: '{.metadata.name}'  
	##############################################################` +
		"\n```" + `
	Suppose your workflow YAML only has a single template as follows.
	$ cat workflow.yaml` +
		"\n\n```yaml" + `
	apiVersion: argoproj.io/v1alpha1
	kind: Workflow
	metadata:
	  generateName: mlops-
	spec:
	  entrypoint: dag
	  templates:
	  - name: MLops
	    inputs:
	      parameters:
	      - name: functions
	    container:
	      imagePullPolicy: IfNotPresent
	      image: mlops-benchmark:latest
	      command: [python]
	      args: ["-m",  "{{inputs.parameters.functions}}"]
	  - name: dag
	    dag:
	      tasks:
	      - name: producer
	        template: MLops
	        arguments:
	          parameters:
	          - name: functions
	            value: producer.py
	      - name: consumer
	        template: MLops
	        dependencies:
	          - producer
	        arguments:
	          parameters:
	          - name: functions
	            value: consumer.py` +
		"\n```" + `
	Suppose only the 'consumer' task need to use vineyard volume,
	you can inject the vineyard volume into the workflow manifest
	with the following command:
	$ vineyardctl inject argo-workflow -f workflow.yaml \
	     --templates="MLops" \	
	     --vineyard-cluster="vineyard-system/vineyardd-sample" \
	     --mount-path="/vineyard/data" \
	     --dag="dag" \
	     --tasks="consumer"
	
	Then the injected manifest will be as follows:` +
		"\n\n```yaml" + `
	apiVersion: argoproj.io/v1alpha1
	kind: Workflow
	metadata:
	  generateName: mlops-
	spec:
	  entrypoint: dag
	  templates:
	  - container:
	      args:
	      - -m
	      - '{{inputs.parameters.functions}}'
	      command:
	      - python
	      image: mlops-benchmark:latest
	      imagePullPolicy: IfNotPresent
	      ############# Injected ############
	      volumeMounts:
	      - name: vineyard-objects
	        mountPath: /vineyard/data
	      ###################################
	    inputs:
	      parameters:
	      - name: functions
	      ############# Injected ############
	      - name: vineyard-objects-name
	      ###################################
	    name: MLops
	    ######################### Injected ########################
	    volumes:
	    - name: vineyard-objects
	      persistentVolumeClaim:
	        claimName: '{{inputs.parameters.vineyard-objects-name}}'
	    ###########################################################
	  - dag:
	      tasks:
	      - arguments:
	          parameters:
	          - name: functions
	            value: producer.py
	          ################################# Injected #################################
	          - name: vineyard-objects-name
	            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
	          ############################################################################
	        dependencies:
	        ##### Injected #####
	        - vineyard-objects
	        ####################
	        name: producer
	        template: MLops
	      - arguments:
	          parameters:
	          - name: functions
	            value: consumer.py
	          ################################# Injected #################################
	          - name: vineyard-objects-name
	            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
	          ############################################################################
	        dependencies:
	        - producer
	        ##### Injected #####
	        - vineyard-objects
	        ####################
	        name: consumer
	        template: MLops
	      ########### Injected ###########
	      - name: vineyard-objects
	        template: vineyard-objects
	      ################################
	    name: dag
	  ################################# Injected #################################
	  - name: vineyard-objects
	    outputs:
	      parameters:
	      - name: vineyard-objects-name
	        valueFrom:
	          jsonPath: '{.metadata.name}'
	    resource:
	      action: create
	      manifest: |-
	        apiVersion: v1
	        kind: PersistentVolumeClaim
	        metadata:
	          name: '{{workflow.name}}-vineyard-objects-pvc'
	        spec:
	          accessModes:
	          - ReadWriteMany
	          resources:
	            requests:
	              storage: 1Mi
	          storageClassName: vineyard-system.vineyardd-sample.csi
	      setOwnerReference: true
	      ############################################################################` +
		"\n```")

	injectArgoWorkflowExample = util.Examples(`
	# Inject the vineyard volumes into the argo workflow
	$ vineyardctl inject argo-workflow -f workflow.yaml \
	     --templates="preprocess-data,train-data" \
	     --vineyard-cluster="vineyard-system/vineyardd-sample" \
	     --mount-path="/vineyard/data" \
	     --dag="dag" \
	     --tasks="preprocess-data,train-data"
	
	# Suppose you only have a single template in the workflow
	# you could set only one template name in the --templates flag
	$ vineyardctl inject argo-workflow -f workflow.yaml \
         --templates="mlops" \
		 --vineyard-cluster="vineyard-system/vineyardd-sample" \
		 --mount-path="/vineyard/data" \
		 --dag="dag" \
		 --tasks="preprocess-data,test-data"
	`)
)

// injectArgoWorkflowCmd inject the vineyard volumes into the argo workflow
var injectArgoWorkflowCmd = &cobra.Command{
	Use:     "argo-workflow",
	Short:   "Inject the vineyard volumes into the argo workflow",
	Long:    injectArgoWorkflowLong,
	Example: injectArgoWorkflowExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		manifest, err := inject()
		if err != nil {
			log.Fatal(err, "Failed to inject the vineyard volumes into the argo workflow")
		}
		if flags.OutputAsFile {
			file := strings.ReplaceAll(flags.ArgoWorkflowFile, ".yaml", "_with_vineyard.yaml")
			if err = util.WriteToFile(file, manifest); err != nil {
				log.Fatal(err, "Failed to write the injected argo workflow to file")
			}
		} else {
			log.Output(manifest)
		}
	},
}

func NewInjectArgoWorkflowCmd() *cobra.Command {
	return injectArgoWorkflowCmd
}

func init() {
	flags.ApplyInjectArgoWorkflowOpts(injectArgoWorkflowCmd)
}

func getArgoWorkflowResource() (string, error) {
	if flags.ArgoWorkflowFile == "" {
		return "", fmt.Errorf("the file name of argo workflow is empty")
	}

	resource, err := util.ReadFromFile(flags.ArgoWorkflowFile)
	if err != nil {
		return resource, errors.Wrap(err, "failed to read the YAML from file")
	}
	return resource, nil
}

func convertTemplateToMap(t interface{}) (map[string]interface{}, error) {
	switch template := t.(type) {
	case map[string]interface{}:
		return template, nil
	case map[interface{}]interface{}:
		result := make(map[string]interface{})
		for k, v := range template {
			result[k.(string)] = v
		}
		return result, nil
	default:
		return nil, fmt.Errorf("failed to convert template to map[string]interface{}")
	}
}

// inject the volumes and volume mounts into the argo workflow template
func injectWorkflowTemplate(workflowYAML string) (*map[string]interface{}, error) {
	var workflow map[string]interface{}
	err := yaml.Unmarshal([]byte(workflowYAML), &workflow)
	if err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal the argo workflow YAML")
	}

	// 1. add the new template "vineyard-objects"
	cluster := strings.ReplaceAll(flags.VineyardCluster, "/", ".")
	vineyardObjectsTemplate := map[string]interface{}{
		"name": "vineyard-objects",
		"resource": map[string]interface{}{
			"action":            "create",
			"setOwnerReference": true,
			"manifest": `apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: '{{workflow.name}}-vineyard-objects-pvc'
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 1Mi
  storageClassName: ` + cluster + `.csi`,
		},
		"outputs": map[string]interface{}{
			"parameters": []map[string]interface{}{
				{
					"name": "vineyard-objects-name",
					"valueFrom": map[string]interface{}{
						"jsonPath": "{.metadata.name}",
					},
				},
			},
		},
	}
	templates, ok := workflow["spec"].(map[interface{}]interface{})["templates"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("failed to convert workflow templates to []interface{}")
	}

	workflow["spec"].(map[interface{}]interface{})["templates"] = append(templates, vineyardObjectsTemplate)

	// 2. add the volumes and volume mounts to the existing template
	templatesExist := make(map[string]bool)
	for _, t := range flags.WorkflowTemplates {
		templatesExist[t] = true
	}

	for i, t := range workflow["spec"].(map[interface{}]interface{})["templates"].([]interface{}) {
		template, err := convertTemplateToMap(t)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert template to map[string]interface{}")
		}

		if templatesExist[template["name"].(string)] {
			// add volume
			volume := map[string]interface{}{
				"name": "vineyard-objects",
				"persistentVolumeClaim": map[string]interface{}{
					"claimName": "{{inputs.parameters.vineyard-objects-name}}",
				},
			}
			if template["volumes"] == nil {
				template["volumes"] = []interface{}{volume}
			} else {
				template["volumes"] = append(template["volumes"].([]interface{}), volume)
			}

			// add volume mounts
			volumeMounts := map[string]interface{}{
				"name":      "vineyard-objects",
				"mountPath": flags.MountPath,
			}
			if template["container"].(map[interface{}]interface{})["volumeMounts"] == nil {
				template["container"].(map[interface{}]interface{})["volumeMounts"] = []interface{}{volumeMounts}
			} else {
				template["container"].(map[interface{}]interface{})["volumeMounts"] = append(
					template["container"].(map[interface{}]interface{})["volumeMounts"].([]interface{}), volumeMounts,
				)
			}

			// add parameters
			parameters := map[string]interface{}{
				"name": "vineyard-objects-name",
			}
			if template["inputs"] == nil {
				template["inputs"] = map[string]interface{}{
					"parameters": []interface{}{parameters},
				}
			} else {
				if template["inputs"].(map[interface{}]interface{})["parameters"] == nil {
					template["inputs"].(map[interface{}]interface{})["parameters"] = []interface{}{parameters}
				} else {
					template["inputs"].(map[interface{}]interface{})["parameters"] = append(
						template["inputs"].(map[interface{}]interface{})["parameters"].([]interface{}), parameters,
					)
				}
			}

			workflow["spec"].(map[interface{}]interface{})["templates"].([]interface{})[i] = template
		}
	}

	return &workflow, nil
}

// inject the dependencies and arguments into the argo workflow dag tasks
func injectWorkflowDagTask(workflow *map[string]interface{}) error {
	tasks := make(map[string]bool)
	for _, t := range flags.Tasks {
		tasks[t] = true
	}
	templates := make(map[string]bool)
	for _, t := range flags.WorkflowTemplates {
		templates[t] = true
	}

	for i, t := range (*workflow)["spec"].(map[interface{}]interface{})["templates"].([]interface{}) {
		template, err := convertTemplateToMap(t)
		if err != nil {
			return errors.Wrap(err, "failed to convert template to map[string]interface{}")
		}
		if template["name"].(string) == flags.Dag {
			// 1. add a new dag task "vineyard-objects" using the template "vineyard-objects"
			template["dag"].(map[interface{}]interface{})["tasks"] = append(
				template["dag"].(map[interface{}]interface{})["tasks"].([]interface{}),
				map[string]interface{}{
					"name":     "vineyard-objects",
					"template": "vineyard-objects",
				},
			)
			newTasks := []map[string]interface{}{}
			for _, t := range template["dag"].(map[interface{}]interface{})["tasks"].([]interface{}) {
				task, err := convertTemplateToMap(t)
				if err != nil {
					return errors.Wrap(err, "failed to convert template to map[string]interface{}")
				}

				// 2. add the dependencies and arguments to the existing dag tasks
				if task["name"] != nil && tasks[task["name"].(string)] || templates[task["template"].(string)] {
					// add dependencies
					if task["dependencies"] == nil {
						task["dependencies"] = []string{
							"vineyard-objects",
						}
					} else {
						task["dependencies"] = append(
							task["dependencies"].([]interface{}),
							"vineyard-objects",
						)
					}

					// add arguments
					parameter := map[string]interface{}{
						"name":  "vineyard-objects-name",
						"value": "{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}",
					}
					if task["arguments"] == nil {
						task["arguments"] = map[string]interface{}{
							"parameters": []interface{}{parameter},
						}
					} else {
						if task["arguments"].(map[interface{}]interface{})["parameters"] == nil {
							task["arguments"].(map[interface{}]interface{})["parameters"] = []interface{}{parameter}
						} else {
							task["arguments"].(map[interface{}]interface{})["parameters"] = append(
								task["arguments"].(map[interface{}]interface{})["parameters"].([]interface{}),
								parameter,
							)
						}
					}
				}
				newTasks = append(newTasks, task)
			}
			template["dag"].(map[interface{}]interface{})["tasks"] = newTasks
			(*workflow)["spec"].(map[interface{}]interface{})["templates"].([]interface{})[i] = template
		}
	}
	return nil
}

func inject() (string, error) {
	resource, err := getArgoWorkflowResource()
	if err != nil {
		return "", err
	}
	workflow, err := injectWorkflowTemplate(resource)
	if err != nil {
		return "", err
	}

	if err = injectWorkflowDagTask(workflow); err != nil {
		return "", errors.Wrap(err, "failed to inject the dependencies and arguments into the argo workflow dag tasks")
	}

	// marshal the argo workflow as YAML
	workflowYAML, err := yaml.Marshal(*workflow)
	if err != nil {
		return "", errors.Wrap(err, "failed to marshal the argo workflow as YAML")
	}

	return string(workflowYAML), nil
}
