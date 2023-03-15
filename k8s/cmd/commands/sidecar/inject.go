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
package sidecar

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"go.uber.org/multierr"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/injector"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

var (
	injectLong = util.LongDesc(`
	Inject the vineyard sidecar container into a workload. You can
	get the injected workload yaml and some etcd yaml from the output.`)

	injectExample = util.Examples(`
	# inject the default vineyard sidecar container into a workload
	vineyardctl inject -f workload.yaml | kubectl apply -f -`)

	// the following label is for vineyard rpc service
	// DefaultSidecarLabelName is the default label name of the vineyard sidecar container
	DefaultSidecarLabelName = "app.kubernetes.io/name"
	// DefaultSidecarLabelValue = "vineyard-sidecar"
	DefaultSidecarLabelValue = "vineyard-sidecar"
)

// injectCmd inject the vineyard sidecar container into a workload
var injectCmd = &cobra.Command{
	Use:     "inject",
	Short:   "Inject the vineyard sidecar container into a workload",
	Long:    injectLong,
	Example: injectExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		resource, err := util.ReadFromFile(flags.WorkloadYaml)
		if err != nil {
			log.Fatal(err, "failed to read the YAML spec from file")
		}

		yamls, err := GetManifestFromTemplate(resource)
		if err != nil {
			log.Fatal(err, "failed to load manifest from template")
		}

		fmt.Println(strings.Join(yamls, "---\n"))
	},
}

// EtcdConfig holds the configuration of etcd
var EtcdConfig k8s.EtcdConfig

func getEtcdConfig() k8s.EtcdConfig {
	return EtcdConfig
}

func GetWorkloadObj(workload string) (*unstructured.Unstructured, error) {
	decoder := util.Deserializer()
	obj, _, err := decoder.Decode([]byte(workload), nil, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to decode the workload")
	}

	proto, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, errors.Wrap(err, "failed to convert the workload to unstructured")
	}
	unstructuredObj := &unstructured.Unstructured{Object: proto}

	// check if the workload is valid
	_, found, err := unstructured.NestedMap(unstructuredObj.Object, "spec", "template")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the template of the workload")
	}
	if !found {
		return nil, errors.Wrap(
			fmt.Errorf("failed to find the template of the workload"),
			"invalid workload",
		)
	}

	return unstructuredObj, nil
}

// SidecarLabelSelector contains the label selector of vineyard sidecar
var SidecarLabelSelector []k8s.ServiceLabelSelector

func getSidecarLabelSelector() []k8s.ServiceLabelSelector {
	return SidecarLabelSelector
}

func GetManifestFromTemplate(workload string) ([]string, error) {
	objects := []*unstructured.Unstructured{}
	manifests := []string{}
	yamls := []string{}

	workloadObj, err := GetWorkloadObj(workload)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get workload object")
	}
	namespace := workloadObj.GetNamespace()

	sidecar, err := buildSidecar(namespace)
	if err != nil {
		return nil, errors.Wrap(err, "failed to build sidecar")
	}

	sidecarManifests, err := templates.GetFilesRecursive("sidecar")
	if err != nil {
		return manifests, errors.Wrap(err, "failed to get sidecar manifests")
	}

	vineyardManifests, err := templates.GetFilesRecursive("vineyardd")
	if err != nil {
		return manifests, errors.Wrap(err, "failed to get vineyardd manifests")
	}

	tmplFunc := map[string]interface{}{
		"getEtcdConfig":           getEtcdConfig,
		"getServiceLabelSelector": getSidecarLabelSelector,
	}

	objs, err := util.BuildObjsFromEtcdManifests(&EtcdConfig, namespace,
		sidecar.Spec.Replicas, sidecar.Spec.VineyardConfig.Image, sidecar, tmplFunc)
	if err != nil {
		return manifests, errors.Wrap(err, "failed to build etcd objects")
	}
	objects = append(objects, objs...)

	// set up the service for etcd
	for _, vf := range vineyardManifests {
		if vf == "vineyardd/etcd-service.yaml" || vf == "vineyardd/service.yaml" {
			obj, err := util.RenderManifestAsObj(vf, sidecar, tmplFunc)
			if err != nil {
				return manifests, errors.Wrap(err,
					fmt.Sprintf("failed to render manifest %s", vf))
			}
			objects = append(objects, obj)
		}
	}

	// set up the vineyard sidecar
	for _, sf := range sidecarManifests {
		obj, err := util.RenderManifestAsObj(sf, sidecar, tmplFunc)
		if err != nil {
			return manifests, errors.Wrap(err,
				fmt.Sprintf("failed to render manifest %s", sf))
		}
		if err := InjectSidecarConfig(sidecar, workloadObj, obj); err != nil {
			return manifests, errors.Wrap(err, "failed to inject the sidecar config")
		}
		objects = append(objects, workloadObj)
	}

	for _, o := range objects {
		ss, err := o.MarshalJSON()
		if err != nil {
			return manifests, errors.Wrap(err, "failed to marshal the unstructuredObj")
		}

		yaml, err := util.ConvertToYaml(string(ss))
		if err != nil {
			return manifests, errors.Wrap(err,
				"failed to convert the unstructuredObj to yaml")
		}
		yamls = append(yamls, yaml)
	}

	return yamls, nil
}

func buildSidecar(namespace string) (*v1alpha1.Sidecar, error) {
	opts := &flags.SidecarOpts
	envs := &flags.VineyardContainerEnvs

	spillPVandPVC := flags.VineyardSpillPVandPVC
	if len(*envs) != 0 {
		vineyardContainerEnvs, err := util.ParseEnvs(*envs)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse envs")
		}
		opts.VineyardConfig.Env = append(opts.VineyardConfig.Env,
			vineyardContainerEnvs...)
	}

	if spillPVandPVC != "" {
		spillPVandPVCJson, err := util.ConvertToJson(spillPVandPVC)
		if err != nil {
			return nil, errors.Wrap(err,
				"failed to convert the pv and pvc of backup to json")
		}
		spillPVSpec, spillPVCSpec, err := util.ParsePVandPVCSpec(spillPVandPVCJson)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pv and pvc of backup")
		}
		opts.VineyardConfig.SpillConfig.PersistentVolumeSpec = *spillPVSpec
		opts.VineyardConfig.SpillConfig.PersistentVolumeClaimSpec = *spillPVCSpec
	}

	sidecar := &v1alpha1.Sidecar{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultSidecarLabelValue,
			Namespace: namespace,
		},
		Spec: *opts,
	}
	return sidecar, nil
}

// InjectSidecarConfig injects the sidecar config into the workload
func InjectSidecarConfig(sidecar *v1alpha1.Sidecar, workloadObj,
	sidecarObj *unstructured.Unstructured,
) error {
	var errList error

	// get the template spec of the workload
	spec := workloadObj.Object["spec"].(map[string]interface{})
	template := spec["template"].(map[string]interface{})
	templateSpec := template["spec"].(map[string]interface{})

	// get the labels of the workload
	metadata := template["metadata"].(map[string]interface{})
	labels := metadata["labels"].(map[string]interface{})

	workloadContainers, err := convertInterfaceToContainers(
		templateSpec["containers"].([]interface{}))
	_ = multierr.Append(errList, err)

	if templateSpec["volumes"] == nil {
		templateSpec["volumes"] = []interface{}{}
	}

	workloadVolumes, err := convertInterfaceToVolumes(
		templateSpec["volumes"].([]interface{}))
	_ = multierr.Append(errList, err)

	// get the containers and volumes of the sidecar
	sidecarSpec := sidecarObj.Object["spec"].(map[string]interface{})
	sidecarContainers, err := convertInterfaceToContainers(
		sidecarSpec["containers"].([]interface{}))
	_ = multierr.Append(errList, err)

	sidecarVolumes, err := convertInterfaceToVolumes(
		sidecarSpec["volumes"].([]interface{}))
	_ = multierr.Append(errList, err)

	injector.InjectSidecar(workloadContainers, workloadVolumes,
		sidecarContainers, sidecarVolumes, sidecar)

	// add labels to the workload for vineyard rpc service
	labels[DefaultSidecarLabelName] = DefaultSidecarLabelValue
	templateSpec["containers"] = workloadContainers
	templateSpec["volumes"] = workloadVolumes
	return err
}

func convertInterfaceToContainers(v []interface{}) (*[]corev1.Container, error) {
	containers := []corev1.Container{}
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	if err = json.Unmarshal(jsonBytes, &containers); err != nil {
		return nil, err
	}
	return &containers, nil
}

func convertInterfaceToVolumes(v []interface{}) (*[]corev1.Volume, error) {
	volumes := []corev1.Volume{}
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	if err = json.Unmarshal(jsonBytes, &volumes); err != nil {
		return nil, err
	}
	return &volumes, nil
}

func NewInjectCmd() *cobra.Command {
	return injectCmd
}

func init() {
	flags.ApplySidecarOpts(injectCmd)
}
