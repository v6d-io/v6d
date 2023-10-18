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
	"os"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

func TestInjectArgoWorkflowCmd(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		expect string
		// cmd flags
		templates       []string
		mountPath       string
		vineyardCluster string
		dag             string
		tasks           []string
	}{
		{
			name:            "each dag task use different templates",
			templates:       []string{"preprocess-data", "train-data"},
			mountPath:       "/vineyard/data",
			vineyardCluster: "vineyard-system/vineyardd-sample",
			dag:             "dag",
			tasks:           []string{"preprocess-data", "train-data"},
			input: `apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: mlops-
spec:
  entrypoint: dag
  templates:
  - name: prepare-data
    container:
      image: prepare-data:latest
      command: [python]
      args: ["/prepare-data.py"]
  - name: preprocess-data
    container:
      image: preprocess-data:latest
      command: [python]
      args: ["/preprocess-data.py"]
  - name: train-data
    container:
      image: train-data:latest
      command: [python]
      args: ["/train-data.py"]
  - name: test-data
    container:
      image: test-data:latest
      command: [python]
      args: ["/test-data.py"]
  - name: dag
    dag:
      tasks:
      - name: prepare-data
        template: prepare-data
      - name: preprocess-data
        template: preprocess-data
        dependencies:
          - prepare-data
      - name: train-data
        template: train-data
        dependencies:
          - preprocess-data
      - name: test-data
        template: test-data
        dependencies:
          - train-data
`,
			expect: `apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: mlops-
spec:
  entrypoint: dag
  templates:
  - container:
      args:
      - /prepare-data.py
      command:
      - python
      image: prepare-data:latest
    name: prepare-data
  - container:
      args:
      - /preprocess-data.py
      command:
      - python
      image: preprocess-data:latest
      volumeMounts:
      - name: vineyard-objects
        mountPath: /vineyard/data
    inputs:
      parameters:
      - name: vineyard-objects-name
    name: preprocess-data
    volumes:
    - name: vineyard-objects
      persistentVolumeClaim:
        claimName: '{{inputs.parameters.vineyard-objects-name}}'
  - container:
      args:
      - /train-data.py
      command:
      - python
      image: train-data:latest
      volumeMounts:
      - name: vineyard-objects
        mountPath: /vineyard/data
    inputs:
      parameters:
      - name: vineyard-objects-name
    name: train-data
    volumes:
    - name: vineyard-objects
      persistentVolumeClaim:
        claimName: '{{inputs.parameters.vineyard-objects-name}}'
  - container:
      args:
      - /test-data.py
      command:
      - python
      image: test-data:latest
    name: test-data
  - dag:
      tasks:
      - name: prepare-data
        template: prepare-data
      - arguments:
          parameters:
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
        dependencies:
        - prepare-data
        - vineyard-objects
        name: preprocess-data
        template: preprocess-data
      - arguments:
          parameters:
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
        dependencies:
        - preprocess-data
        - vineyard-objects
        name: train-data
        template: train-data
      - dependencies:
        - train-data
        name: test-data
        template: test-data
      - name: vineyard-objects
        template: vineyard-objects
    name: dag
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
      setOwnerReference: true`,
		},
		{
			name:            "each dag task use the same template",
			templates:       []string{"MLops"},
			mountPath:       "/vineyard/data",
			vineyardCluster: "vineyard-system/vineyardd-sample",
			dag:             "dag",
			tasks:           []string{"preprocess-data", "train-data"},
			input: `apiVersion: argoproj.io/v1alpha1
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
      - name: prepare-data
        template: MLops
        arguments:
          parameters:
          - name: functions
            value: prepare-data.py
      - name: preprocess-data
        template: MLops
        dependencies:
          - prepare-data
        arguments:
          parameters:
          - name: functions
            value: preprocess-data.py
      - name: train-data
        template: MLops
        dependencies:
          - preprocess-data
        arguments:
          parameters:
          - name: functions
            value: train-data.py
      - name: test-data
        template: MLops
        dependencies:
          - train-data
        arguments:
          parameters:
          - name: functions
            value: test-data.py
`,
			expect: `apiVersion: argoproj.io/v1alpha1
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
      volumeMounts:
      - name: vineyard-objects
        mountPath: /vineyard/data
    inputs:
      parameters:
      - name: functions
      - name: vineyard-objects-name
    name: MLops
    volumes:
    - name: vineyard-objects
      persistentVolumeClaim:
        claimName: '{{inputs.parameters.vineyard-objects-name}}'
  - dag:
      tasks:
      - arguments:
          parameters:
          - name: functions
            value: prepare-data.py
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
        dependencies:
        - vineyard-objects
        name: prepare-data
        template: MLops
      - arguments:
          parameters:
          - name: functions
            value: preprocess-data.py
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
        dependencies:
        - prepare-data
        - vineyard-objects
        name: preprocess-data
        template: MLops
      - arguments:
          parameters:
          - name: functions
            value: train-data.py
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
        dependencies:
        - preprocess-data
        - vineyard-objects
        name: train-data
        template: MLops
      - arguments:
          parameters:
          - name: functions
            value: test-data.py
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
        dependencies:
        - train-data
        - vineyard-objects
        name: test-data
        template: MLops
      - name: vineyard-objects
        template: vineyard-objects
    name: dag
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
`,
		},
	}
	// Create a temporary argo workflow file
	file, err := os.CreateTemp("", "workflow.yaml")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}
	defer file.Close()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _ = file.WriteString(tt.input)
			flags.ArgoWorkflowFile = file.Name()
			flags.WorkflowTemplates = tt.templates
			flags.MountPath = tt.mountPath
			flags.VineyardCluster = tt.vineyardCluster
			flags.Dag = tt.dag
			flags.Tasks = tt.tasks
			output := util.CaptureCmdOutput(injectArgoWorkflowCmd)
			var outputMap, expectMap map[string]interface{}
			if err := yaml.Unmarshal([]byte(output), &outputMap); err != nil {
				t.Errorf("unmarshal output YAML error: %v", err)
			}
			if err := yaml.Unmarshal([]byte(tt.expect), &expectMap); err != nil {
				t.Errorf("unmarshal expect YAML error: %v", err)
			}

			if !reflect.DeepEqual(outputMap, expectMap) {
				t.Errorf("%v error: injectArgoWorkflowCmd() = %v, want %v", tt.name, outputMap, expectMap)
			}
		})
	}
	// Remove the file when the test is done
	if err := os.Remove(file.Name()); err != nil {
		t.Fatalf("Failed to remove temporary file: %v", err)
	}
}
