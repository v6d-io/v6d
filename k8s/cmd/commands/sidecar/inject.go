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
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

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
	unstructuredObj, err := util.ParseManifestToObject(workload)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse the workload")
	}

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

// sidecarLabelSelector contains the label selector of vineyard sidecar
var sidecarLabelSelector []k8s.ServiceLabelSelector

func getSidecarLabelSelector() []k8s.ServiceLabelSelector {
	return sidecarLabelSelector
}

func GetManifestFromTemplate(workload string) ([]string, error) {
	objects := make([]*unstructured.Unstructured, 0)
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

	tmplFunc := map[string]interface{}{
		"getEtcdConfig":           getEtcdConfig,
		"getServiceLabelSelector": getSidecarLabelSelector,
	}

	objs, err := util.BuildObjsFromEtcdManifests(&EtcdConfig, namespace,
		sidecar.Spec.Replicas, sidecar.Spec.Vineyard.Image, sidecar, tmplFunc)
	if err != nil {
		return manifests, errors.Wrap(err, "failed to build etcd objects")
	}
	objects = append(objects, objs...)

	// set up the service for etcd
	files := []string{"vineyardd/etcd-service.yaml", "vineyardd/service.yaml"}
	objs, err = util.BuildObjsFromVineyarddManifests(files, sidecar, tmplFunc)
	if err != nil {
		return manifests, errors.Wrap(err, "failed to build vineyardd objects")
	}
	objects = append(objects, objs...)

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
	selector := DefaultSidecarLabelName + "=" + DefaultSidecarLabelValue
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
	flags.ApplySidecarOpts(injectCmd)
}
