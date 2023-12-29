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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/injector"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// OutputManifests contains all output json string of the injection
type OutputManifests struct {
	// Workload is the json string of the injected workload
	// which contains the vineyard sidecar container
	Workload string `json:"workload"`

	// RPCService is the json string of the rpc service of the
	// injected vineyard sidecar container
	RPCService string `json:"rpc_service"`

	// EtcdService is the json string of the etcd service of the
	// injected vineyard sidecar container, which is used to connect
	// the other vineyard sidecar containers when the workload uses
	// the internal etcd cluster or connects to the external etcd cluster
	// when the workload uses the external etcd cluster
	EtcdService string `json:"etcd_service"`

	// EtcdInternalServiceJSON is the json string of the etcd service
	// which is used as the internal service to build a etcd cluster
	// for vineyard sidecar container
	EtcdInternalService []string `json:"etcd_internal_service"`

	// EtcdPodJSON is the json string of the etcd pod
	// which is used as the external etcd cluster
	EtcdPod []string `json:"etcd_pod"`
}

var (
	// JSONFormat is the json format
	JSONFormat = "json"
	// YAMLFormat is the yaml format
	YAMLFormat = "yaml"

	injectLong = util.LongDesc(`
	Inject the vineyard sidecar container into a workload. You can
	input a workload yaml or a workload json and then get the injected
	workload and some etcd manifests from the output. The workload can
	be a pod or a deployment or a statefulset, etc.

	The output is a set of manifests that includes the injected workload,
	the rpc service, the etcd service and the etcd cluster(e.g. several
	pods and services). 
	
	If you have a pod yaml: ` +
		"\n\n```yaml" + `
	apiVersion: v1
	kind: Pod
	metadata:
	  name: python
	spec:
	  containers:
	  - name: python
	    image: python:3.10
	    command: ["python", "-c", "import time; time.sleep(100000)"]` +
		"\n```" + `
	Then, you can use the following command to inject the vineyard sidecar

	$ vineyardctl inject -f pod.yaml
	
	After running the command, the output is as follows:` +
		"\n\n```yaml" + `
	apiVersion: v1
	kind: Pod
	metadata:
	  labels:
	    app.vineyard.io/name: vineyard-sidecar
	    app.vineyard.io/role: etcd
	    etcd_node: vineyard-sidecar-etcd-0
	  name: vineyard-sidecar-etcd-0
	  namespace: null
	  ownerReferences: []
	spec:
	  containers:
	  - command:
	    - etcd
	    - --name
	    - vineyard-sidecar-etcd-0
	    - --initial-advertise-peer-urls
	    - http://vineyard-sidecar-etcd-0:2380
	    - --advertise-client-urls
	    - http://vineyard-sidecar-etcd-0:2379
	    - --listen-peer-urls
	    - http://0.0.0.0:2380
	    - --listen-client-urls
	    - http://0.0.0.0:2379
	    - --initial-cluster
	    - vineyard-sidecar-etcd-0=http://vineyard-sidecar-etcd-0:2380
	    - --initial-cluster-state
	    - new
	    image: vineyardcloudnative/vineyardd:latest
	    name: etcd
	    ports:
	    - containerPort: 2379
	      name: client
	      protocol: TCP
	    - containerPort: 2380
	      name: server
	      protocol: TCP
	  restartPolicy: Always
	---
	apiVersion: v1
	kind: Service
	metadata:
	  labels:
	    etcd_node: vineyard-sidecar-etcd-0
	  name: vineyard-sidecar-etcd-0
	  namespace: null
	  ownerReferences: []
	spec:
	  ports:
	  - name: client
	    port: 2379
	    protocol: TCP
	    targetPort: 2379
	  - name: server
	    port: 2380
	    protocol: TCP
	    targetPort: 2380
	  selector:
	    app.vineyard.io/role: etcd
	    etcd_node: vineyard-sidecar-etcd-0
	---
	apiVersion: v1
	kind: Service
	metadata:
	  name: vineyard-sidecar-etcd-service
	  namespace: null
	  ownerReferences: []
	spec:
	  ports:
	  - name: etcd-for-vineyard-port
	    port: 2379
	    protocol: TCP
	    targetPort: 2379
	  selector:
	    app.vineyard.io/name: vineyard-sidecar
	    app.vineyard.io/role: etcd
	---
	apiVersion: v1
	kind: Service
	metadata:
	  labels:
	    app.vineyard.io/name: vineyard-sidecar
	  name: vineyard-sidecar-rpc
	  namespace: null
	  ownerReferences: []
	spec:
	  ports:
	  - name: vineyard-rpc
	    port: 9600
	    protocol: TCP
	  selector:
	    app.vineyard.io/name: vineyard-sidecar
	    app.vineyard.io/role: vineyardd
	  type: ClusterIP
	---
	apiVersion: v1
	kind: Pod
	metadata:
	  creationTimestamp: null
	  labels:
	    app.vineyard.io/name: vineyard-sidecar
	    app.vineyard.io/role: vineyardd
	  name: python
	  ownerReferences: []
	spec:
	  containers:
	  - command:
	    - python
	    - -c
	    - while [ ! -e /var/run/vineyard.sock ]; do sleep 1; done;import time; time.sleep(100000)
	    env:
	    - name: VINEYARD_IPC_SOCKET
	      value: /var/run/vineyard.sock
	    image: python:3.10
	    name: python
	    resources: {}
	    volumeMounts:
	    - mountPath: /var/run
	      name: vineyard-socket
	  - command:
	    - /bin/bash
	    - -c
	    - |
	      /usr/local/bin/vineyardd --sync_crds true --socket /var/run/vineyard.sock --size \
	      --stream_threshold 80 --etcd_cmd etcd --etcd_prefix /vineyard --etcd_endpoint http://vineyard-sidecar-etcd-service:2379
	    env:
	    - name: VINEYARDD_UID
	      value: null
	    - name: VINEYARDD_NAME
	      value: vineyard-sidecar
	    - name: VINEYARDD_NAMESPACE
	      value: null
	    image: vineyardcloudnative/vineyardd:latest
	    imagePullPolicy: IfNotPresent
	    name: vineyard-sidecar
	    ports:
	    - containerPort: 9600
	      name: vineyard-rpc
	      protocol: TCP
	    resources:
	      limits: null
	      requests: null
	    securityContext: {}
	    volumeMounts:
	    - mountPath: /var/run
	      name: vineyard-socket
	  volumes:
	  - emptyDir: {}
	    name: vineyard-socket
	status: {}` +
		"\n```" + `

	Next, we will introduce a simple example to show the injection with
	the apply-resources flag.

	Assume you have the following workload yaml:` +
		"\n\n```yaml" + `
	apiVersion: apps/v1
	kind: Deployment
	metadata:
	  name: nginx-deployment
	    # Notice, you must set the namespace here
	  namespace: vineyard-job
	spec:
	  selector:
	    matchLabels:
	      app: nginx
	  template:
	    metadata:
	      labels:
	        app: nginx
	    spec:
	      containers:
	      - name: nginx
	        image: nginx:1.14.2
	        ports:
	        - containerPort: 80` +
		"\n```" + `

	Then, you can use the following command to inject the vineyard sidecar
	which means that all resources will be created during the injection except
	the workload itself. The workload should be created by users.

	$ vineyardctl inject -f workload.yaml --apply-resources

	After running the command, the main output(removed some unnecessary fields)
	is as follows:` +
		"\n\n```yaml" + `
	apiVersion: apps/v1
	kind: Deployment
	metadata:
	  creationTimestamp: null
	  name: nginx-deployment
	  namespace: vineyard-job
	spec:
	  selector:
	    matchLabels:
	      app: nginx
	template:
	  metadata:
	  labels:
	    app: nginx
	    # the default sidecar name is vineyard-sidecar
	    app.vineyard.io/name: vineyard-sidecar
	  spec:
	    containers:
	    - command: null
	      image: nginx:1.14.2
	      name: nginx
	      ports:
	      - containerPort: 80
	      volumeMounts:
	      - mountPath: /var/run
	        name: vineyard-socket
	    - command:
	      - /bin/bash
	      - -c
	      - |
	        /usr/local/bin/vineyardd --sync_crds true --socket /var/run/vineyard.sock \
	        --stream_threshold 80 --etcd_cmd etcd --etcd_prefix /vineyard \
	        --etcd_endpoint http://vineyard-sidecar-etcd-service:2379
	      env:
	      - name: VINEYARDD_UID
	        value: null
	      - name: VINEYARDD_NAME
	        value: vineyard-sidecar
	      - name: VINEYARDD_NAMESPACE
	        value: vineyard-job
	      image: vineyardcloudnative/vineyardd:latest
	      imagePullPolicy: IfNotPresent
	      name: vineyard-sidecar
	      ports:
	      - containerPort: 9600
	        name: vineyard-rpc
	        protocol: TCP
	      volumeMounts:
	      - mountPath: /var/run
	        name: vineyard-socket
	    volumes:
	    - emptyDir: {}
	      name: vineyard-socket` +
		"\n```" + `

	The sidecar template can be accessed from the following link:
	https://github.com/v6d-io/v6d/blob/main/k8s/pkg/templates/sidecar/injection-template.yaml
	also you can get some inspiration from the doc link:
	https://v6d.io/notes/cloud-native/vineyard-operator.html#installing-vineyard-as-sidecar`)

	injectExample = util.Examples(`
	# use json format to output the injected workload
	# notice that the output is a json string of all manifests
	# it looks like:
	# {
	#   "workload": "workload json string",
	#   "rpc_service": "rpc service json string",
	#   "etcd_service": "etcd service json string",
	#   "etcd_internal_service": [
	#     "etcd internal service json string 1",
	#     "etcd internal service json string 2",
	#     "etcd internal service json string 3"
	#   ],
	#   "etcd_pod": [
	#     "etcd pod json string 1",
	#     "etcd pod json string 2",
	#     "etcd pod json string 3"
	#   ]
	# }
	vineyardctl inject -f workload.yaml -o json

	# inject the default vineyard sidecar container into a workload
	# output all injected manifests and then deploy them
	vineyardctl inject -f workload.yaml | kubectl apply -f -

	# if you only want to get the injected workload yaml rather than
	# all manifests that includes the etcd cluster and the rpc service,
	# you can enable the apply-resources and then the manifests will be
	# created during the injection, finally you will get the injected
	# workload yaml
	vineyardctl inject -f workload.yaml --apply-resources`)
)

// injectCmd inject the vineyard sidecar container into a workload
var injectCmd = &cobra.Command{
	Use:     "inject",
	Short:   "Inject the vineyard sidecar container into a workload",
	Long:    injectLong,
	Example: injectExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		if err := validateFormat(flags.OutputFormat); err != nil {
			log.Fatal(err, "invalid output format")
		}

		resource, err := getWorkloadResource()
		if err != nil {
			log.Fatal(err, "failed to get the workload resource")
		}

		manifests, err := GetManifestFromTemplate(resource)
		if err != nil {
			log.Fatal(err, "failed to load manifest from template")
		}

		if flags.ApplyResources {
			if err := deployDuringInjection(&manifests); err != nil {
				log.Fatal(err, "failed to deploy the manifests during injection")
			}
		}

		if err := outputInjectedResult(manifests); err != nil {
			log.Fatal(err, "failed to output the injected result")
		}
	},
}

// validateFormat checks the format of the output
func validateFormat(format string) error {
	if format != YAMLFormat && format != JSONFormat {
		return errors.New("the output format must be yaml or json")
	}
	return nil
}

// getWorkloadResource returns the workload resource from the input
// return a yaml string
func getWorkloadResource() (string, error) {
	var (
		resource string
		err      error
	)
	if flags.WorkloadResource != "" && flags.WorkloadYaml != "" {
		return "", errors.New("cannot specify both workload resource and workload yaml")
	}
	if flags.WorkloadResource != "" {
		// convert the json string to yaml string
		resource, err = util.ConvertToYaml(flags.WorkloadResource)
		if err != nil {
			return "", errors.Wrap(err, "failed to convert the workload resource to yaml")
		}
		return resource, nil
	}
	resource, err = util.ReadFromFile(flags.WorkloadYaml)
	if err != nil {
		return resource, errors.Wrap(err, "failed to read the YAML from file")
	}
	return resource, nil
}

// GetWorkloadObj returns the unstructured object of the workload
func GetWorkloadObj(workload string) (*unstructured.Unstructured, error) {
	unstructuredObj, err := util.ParseManifestToObject(workload)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse the workload")
	}

	return unstructuredObj, nil
}

// GetManifestFromTemplate returns the manifests from the template
func GetManifestFromTemplate(workload string) (OutputManifests, error) {
	var om OutputManifests
	var etcdConfig k8s.EtcdConfig

	ownerRef, err := util.ParseOwnerRef(flags.OwnerReference)
	if err != nil {
		return om, errors.Wrap(err, "failed to get owner reference from input")
	}

	workloadObj, err := GetWorkloadObj(workload)
	if err != nil {
		return om, errors.Wrap(err, "failed to get workload object")
	}
	namespace := workloadObj.GetNamespace()

	sidecar, err := buildSidecar(namespace)
	if err != nil {
		return om, errors.Wrap(err, "failed to build sidecar")
	}

	sidecarManifests, err := templates.GetFilesRecursive("sidecar")
	if err != nil {
		return om, errors.Wrap(err, "failed to get sidecar manifests")
	}

	tmplFunc := map[string]interface{}{
		"getEtcdConfig": func() k8s.EtcdConfig {
			return etcdConfig
		},
		"toYaml": func(v interface{}) string {
			bs, err := yaml.Marshal(v)
			if err != nil {
				log.Error(err, "failed to marshal object %v to yaml", v)
				return ""
			}
			return string(bs)
		},
		"indent": func(spaces int, s string) string {
			prefix := strings.Repeat(" ", spaces)
			return prefix + strings.Replace(s, "\n", "\n"+prefix, -1)
		},
	}

	podObjs, svcObjs, err := util.BuildObjsFromEtcdManifests(&etcdConfig,
		flags.SidecarName, namespace, sidecar.Spec.Replicas,
		sidecar.Spec.Vineyard.Image, sidecar, tmplFunc)
	if err != nil {
		return om, errors.Wrap(err, "failed to build etcd objects")
	}
	for i := range podObjs {
		podObjs[i].SetOwnerReferences(ownerRef)
		ss, err := podObjs[i].MarshalJSON()
		if err != nil {
			return om, errors.Wrap(err, "failed to marshal the unstructuredObj")
		}
		om.EtcdPod = append(om.EtcdPod, string(ss))
	}

	for i := range svcObjs {
		svcObjs[i].SetOwnerReferences(ownerRef)
		ss, err := svcObjs[i].MarshalJSON()
		if err != nil {
			return om, errors.Wrap(err, "failed to marshal the unstructuredObj")
		}
		om.EtcdInternalService = append(om.EtcdInternalService, string(ss))
	}

	// set up the service for etcd
	files := []string{"vineyardd/etcd-service.yaml"}
	objs, err := util.BuildObjsFromVineyarddManifests(files, sidecar, tmplFunc)
	if err != nil {
		return om, errors.Wrap(err, "failed to build vineyardd objects")
	}
	objs[0].SetOwnerReferences(ownerRef)
	ss, err := objs[0].MarshalJSON()
	if err != nil {
		return om, errors.Wrap(err, "failed to marshal the unstructuredObj")
	}
	om.EtcdService = string(ss)

	// set up the rpc service for vineyardd
	files = []string{"vineyardd/service.yaml"}
	objs, err = util.BuildObjsFromVineyarddManifests(files, sidecar, tmplFunc)
	if err != nil {
		return om, errors.Wrap(err, "failed to build vineyardd objects")
	}
	objs[0].SetOwnerReferences(ownerRef)
	ss, err = objs[0].MarshalJSON()
	if err != nil {
		return om, errors.Wrap(err, "failed to marshal the unstructuredObj")
	}
	om.RPCService = string(ss)

	// set up the vineyard sidecar
	for _, sf := range sidecarManifests {
		obj, err := util.RenderManifestAsObj(sf, sidecar, tmplFunc)
		if err != nil {
			return om, errors.Wrap(err,
				fmt.Sprintf("failed to render manifest %s", sf))
		}
		if err := InjectSidecarConfig(sidecar, workloadObj, obj); err != nil {
			return om, errors.Wrap(err, "failed to inject the sidecar config")
		}
		workloadObj.SetOwnerReferences(ownerRef)
		ss, err := workloadObj.MarshalJSON()
		if err != nil {
			return om, errors.Wrap(err, "failed to marshal the unstructuredObj")
		}
		om.Workload = string(ss)
	}

	return om, nil
}

func parseManifestsAsYAML(om OutputManifests) ([]string, error) {
	var results []string

	if len(om.EtcdPod) != 0 {
		for _, m := range om.EtcdPod {
			output, err := util.ConvertToYaml(m)
			if err != nil {
				return nil, errors.Wrap(err, "failed to convert EtcdPodJSON to yaml")
			}
			results = append(results, output)
		}
	}
	if len(om.EtcdInternalService) != 0 {
		for _, m := range om.EtcdInternalService {
			output, err := util.ConvertToYaml(m)
			if err != nil {
				return nil, errors.Wrap(err, "failed to convert EtcdInternalServiceJSON to yaml")
			}
			results = append(results, output)
		}
	}
	if om.EtcdService != "" {
		output, err := util.ConvertToYaml(om.EtcdService)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert EtcdServiceJSON to yaml")
		}
		results = append(results, output)
	}
	if om.RPCService != "" {
		output, err := util.ConvertToYaml(om.RPCService)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert RPCServiceJSON to yaml")
		}
		results = append(results, output)
	}
	if om.Workload != "" {
		output, err := util.ConvertToYaml(om.Workload)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert WorkloadJSON to yaml")
		}
		results = append(results, output)
	}
	return results, nil
}

// deployDuringInjection deploys the manifests including the etcd cluster and the rpc service
func deployDuringInjection(om *OutputManifests) error {
	jsons := []string{om.EtcdService, om.RPCService}
	jsons = append(jsons, om.EtcdPod...)
	jsons = append(jsons, om.EtcdInternalService...)
	// set up the several jsons to nil to avoid the output
	om.EtcdPod = nil
	om.EtcdInternalService = nil
	om.EtcdService = ""
	om.RPCService = ""

	client := util.KubernetesClient()
	// convert the manifest to unstructured object
	for _, json := range jsons {
		manifest, err := util.ConvertToYaml(json)
		if err != nil {
			return errors.Wrap(err, "failed to convert json to yaml")
		}
		obj, err := util.ParseManifestToObject(manifest)
		if err != nil {
			return errors.Wrap(err, "failed to parse the manifest to object")
		}

		// create the object during the injection
		if err := util.CreateIfNotExists(client, obj); err != nil {
			return errors.Wrap(err, "failed to create the object during the injection")
		}
	}

	return nil
}

func outputInjectedResult(om OutputManifests) error {
	if flags.OutputFormat == JSONFormat {
		outputJSON, _ := json.Marshal(om)
		log.Output(string(outputJSON))
		return nil
	}
	yamls, err := parseManifestsAsYAML(om)
	if err != nil {
		return errors.Wrap(err, "failed to parse manifests as yaml")
	}
	log.Output(strings.Join(yamls, "---\n"))
	return nil
}

func buildSidecar(namespace string) (*v1alpha1.Sidecar, error) {
	opts := &flags.SidecarOpts
	envs := &flags.VineyardContainerEnvs

	if len(*envs) != 0 {
		vineyardContainerEnvs, err := util.ParseEnvs(*envs)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse envs")
		}
		opts.Vineyard.Env = append(opts.Vineyard.Env,
			vineyardContainerEnvs...)
	}

	spillPVandPVC := flags.VineyardSpillPVandPVC
	if spillPVandPVC != "" {
		pv, pvc, err := util.GetPVAndPVC(spillPVandPVC)
		if err != nil {
			return nil, errors.Wrap(err, "failed to get pv and pvc of spill config")
		}
		opts.Vineyard.Spill.PersistentVolumeSpec = *pv
		opts.Vineyard.Spill.PersistentVolumeClaimSpec = *pvc
	}

	sidecar := &v1alpha1.Sidecar{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.SidecarName,
			Namespace: namespace,
		},
		Spec: *opts,
	}
	if flags.VineyardSecurityContext != "" {
		securityContext, err := util.ParseSecurityContext(flags.VineyardSecurityContext)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse security context of vineyard sidecar container")
		}
		sidecar.Spec.SecurityContext = *securityContext
	}
	if flags.VineyardVolume != "" {
		volumes, err := util.ParseVolume(flags.VineyardVolume)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse volumes of vineyard sidecar container")
		}
		sidecar.Spec.Volumes = *volumes
	}
	if flags.VineyardVolumeMount != "" {
		volumeMounts, err := util.ParseVolumeMount(flags.VineyardVolumeMount)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse volume mounts of vineyard sidecar container")
		}
		sidecar.Spec.VolumeMounts = *volumeMounts
	}
	return sidecar, nil
}

// InjectSidecarConfig injects the sidecar config into the workload
func InjectSidecarConfig(sidecar *v1alpha1.Sidecar, workloadObj,
	sidecarObj *unstructured.Unstructured,
) error {
	selector1 := labels.VineyardAppLabel + "=" + flags.SidecarName
	selector2 := labels.VineyardRoleLabel + "=" + "vineyardd"
	selector := selector1 + "," + selector2
	err := injector.InjectSidecar(workloadObj, sidecarObj, sidecar, selector)
	if err != nil {
		return errors.Wrap(err, "failed to inject the sidecar")
	}

	return nil
}

func NewInjectCmd() *cobra.Command {
	return injectCmd
}

func init() {
	injectCmd.AddCommand(NewInjectArgoWorkflowCmd())
	flags.ApplySidecarOpts(injectCmd)
}
