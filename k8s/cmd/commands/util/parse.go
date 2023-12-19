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
package util

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ghodss/yaml"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"

	"github.com/pkg/errors"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ConvertToJson check whether the string is json type and
// converts the string to json type if not
func ConvertToJson(str string) (string, error) {
	result := make(map[string]interface{})
	err := json.Unmarshal([]byte(str), &result)
	if err != nil {
		json, err := yaml.YAMLToJSON([]byte(str))
		if err != nil {
			return "", errors.Wrap(err, "failed to convert to json")
		}
		return string(json), nil
	}
	return str, nil
}

// ConvertToYaml converts the json string to yaml type
func ConvertToYaml(str string) (string, error) {
	y, err := yaml.JSONToYAML([]byte(str))
	if err != nil {
		return "", errors.Wrap(err, "failed to convert to yaml")
	}

	return string(y), nil
}

// ParseEnvs parse the string to container's environment variables
func ParseEnvs(envArray []string) ([]corev1.EnvVar, error) {
	envs := make([]corev1.EnvVar, 0, len(envArray))
	for _, env := range envArray {
		i := strings.Index(env, "=")
		if i == -1 {
			return envs, errors.New("invalid env string")
		}
		name := env[:i]
		value := env[i+1:]
		if name == "" {
			return envs, errors.New("the name of env can not be empty")
		}
		if value == "" {
			return envs, errors.New("the value of env can not be empty")
		}
		envs = append(envs, corev1.EnvVar{
			Name:  env,
			Value: env,
		})
	}
	return envs, nil
}

// ParsePVSpec parse the json string to corev1.PersistentVolumeSpec
func ParsePVSpec(pvspec string) (*corev1.PersistentVolumeSpec, error) {
	// add the spec field to the pvspec string
	pvspec = `{"spec":` + pvspec + `}`

	decoder := Deserializer()
	obj, _, err := decoder.Decode(
		[]byte(pvspec),
		&schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolume"},
		nil,
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to decode the pvspec string to PV Spec")
	}

	pv := obj.(*corev1.PersistentVolume)
	return &pv.Spec, nil
}

// ParsePVCSpec parse the json string to corev1.PersistentVolumeClaimSpec
func ParsePVCSpec(pvcspec string) (*corev1.PersistentVolumeClaimSpec, error) {
	// add the spec field to the pvcspec string
	pvcspec = `{"spec":` + pvcspec + `}`

	decoder := Deserializer()
	obj, _, err := decoder.Decode(
		[]byte(pvcspec),
		&schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolumeClaim"},
		nil,
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to decode the pvspec string to PV Spec")
	}

	pvc := obj.(*corev1.PersistentVolumeClaim)
	return &pvc.Spec, nil
}

// ParsePVandPVCSpec parse the json string to
// corev1.PersistentVolumeSpec and corev1.PersistentVolumeClaimSpec
func ParsePVandPVCSpec(
	PvAndPvcJSON string,
) (*corev1.PersistentVolumeSpec, *corev1.PersistentVolumeClaimSpec, error) {
	var result map[string]interface{}
	err := json.Unmarshal([]byte(PvAndPvcJSON), &result)
	if err != nil {
		return nil, nil, err
	}

	pvSpecStr, err := json.Marshal(result["pv-spec"])
	if err != nil {
		return nil, nil, err
	}

	pvcSpecStr, err := json.Marshal(result["pvc-spec"])
	if err != nil {
		return nil, nil, err
	}

	pvSpec, err := ParsePVSpec(string(pvSpecStr))
	if err != nil {
		return nil, nil, err
	}
	pvcSpec, err := ParsePVCSpec(string(pvcSpecStr))
	if err != nil {
		return nil, nil, err
	}
	return pvSpec, pvcSpec, nil
}

// GetPVAndPVC parse the pVAndPVC string to corev1.PersistentVolumeSpec
// and corev1.PersistentVolumeClaimSpec, then return the relevant fields.
func GetPVAndPVC(pVAndPVC string) (*corev1.PersistentVolumeSpec,
	*corev1.PersistentVolumeClaimSpec, error) {
	var pv *corev1.PersistentVolumeSpec
	var pvc *corev1.PersistentVolumeClaimSpec
	if pVAndPVC != "" {
		json, err := ConvertToJson(pVAndPVC)
		if err != nil {
			return pv, pvc, errors.Wrap(err,
				"failed to convert the pv and pvc to json")
		}
		pv, pvc, err = ParsePVandPVCSpec(json)
		if err != nil {
			return pv, pvc, errors.Wrap(err, "failed to parse the pv and pvc")
		}
	}
	return pv, pvc, nil
}

// ParseOwnerRef parse the string to metav1.OwnerReference
func ParseOwnerRef(of string) ([]metav1.OwnerReference, error) {
	ownerRef := []metav1.OwnerReference{}

	if of != "" {
		ownerRef = make([]metav1.OwnerReference, 1)
		if err := json.Unmarshal([]byte(of), &ownerRef); err != nil {
			return nil, errors.Wrap(err, "failed to unmarshal the owner reference")
		}
	}

	return ownerRef, nil
}

//	ParseVineyardClusters parse the []string to nested []{
//		"namespace": "vineyard-cluster-namespace",
//	    "name": "vineyard-cluster-name",
//	}
func ParseVineyardClusters(clusters []string) (*[]v1alpha1.VineyardClusters, error) {
	vineyardClusters := make([]v1alpha1.VineyardClusters, 0)
	for i := range clusters {
		s := strings.Split(clusters[i], "/")
		if len(s) != 2 {
			return nil, errors.Wrap(fmt.Errorf("invalid vineyard cluster %s", clusters[i]), "parse vineyard cluster")
		}
		vineyardClusters = append(vineyardClusters, v1alpha1.VineyardClusters{
			Namespace: s[0],
			Name:      s[1],
		})
	}
	return &vineyardClusters, nil
}

// ParseVolume parse the json string to corev1.Volume
func ParseVolume(volumeJSON string) (*[]corev1.Volume, error) {
	var volume []corev1.Volume
	err := json.Unmarshal([]byte(volumeJSON), &volume)
	if err != nil {
		return nil, err
	}
	return &volume, nil
}

// ParseVolumeMount parse the json string to corev1.VolumeMount
func ParseVolumeMount(volumeMountJSON string) (*[]corev1.VolumeMount, error) {
	var volumeMount []corev1.VolumeMount
	err := json.Unmarshal([]byte(volumeMountJSON), &volumeMount)
	if err != nil {
		return nil, err
	}
	return &volumeMount, nil
}

// ParseSecurityContext parse the json string to corev1.SecurityContext
func ParseSecurityContext(securityContextJSON string) (*corev1.SecurityContext, error) {
	var securityContext corev1.SecurityContext
	err := json.Unmarshal([]byte(securityContextJSON), &securityContext)
	if err != nil {
		return nil, err
	}
	return &securityContext, nil
}
