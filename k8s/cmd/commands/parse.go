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
package commands

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/yaml"

	corev1 "k8s.io/api/core/v1"
)

// ParseEnvs parse the string to container's environment variables
func ParseEnvs(envArray []string) ([]corev1.EnvVar, error) {
	envs := make([]corev1.EnvVar, 0, len(envArray))
	for _, env := range envArray {
		i := strings.Index(env, "=")
		if i == -1 {
			return envs, fmt.Errorf("invalid env string")
		}
		name := env[:i]
		value := env[i+1:]
		if name == "" {
			return envs, fmt.Errorf("the name of env can not be empty")
		}
		if value == "" {
			return envs, fmt.Errorf("the value of env can not be empty")
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
	//pvSpec := &corev1.PersistentVolumeSpec{}
	fmt.Println("pvspec is: ", pvspec)
	pvspec = "Spec:PersistentVolumeSpec{capacity: \"1Gi\",}"
	//pv := &corev1.PersistentVolume{}
	decode := serializer.NewCodecFactory(CmdScheme).UniversalDeserializer().Decode

	/*if err := json.Unmarshal([]byte(pvspec), &pvSpec); err != nil {
		return pvSpec, fmt.Errorf("failed to unmarshal the pvspec string to PV Spec: %v", err)
	}
	fmt.Println(pvSpec)*/

	obj, _, err := decode([]byte(pvspec), &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolume"}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode the pvspec string to PV Spec: %v", err)
	}
	fmt.Println(obj)

	/*pvbyte, err := json.Marshal(pv)
	if err != nil {
		fmt.Println("error is: ", err)
	}
	fmt.Println(string(pvbyte))*/

	/*if err := pvSpec.Unmarshal([]byte(pvspec)); err != nil {
		fmt.Println("error is: ", err)
	}*/
	//fmt.Println(pv)
	/*if err := json.Unmarshal([]byte(pvspec), &pvSpec); err != nil {
		return pvSpec, fmt.Errorf("failed to unmarshal the pvspec string to PV Spec: %v", err)
	}*/
	return nil, nil
	//return pvSpec, nil
}

// parse the json string to corev1.PersistentVolumeClaimSpec
// ParsePVCSpec parse the json string to corev1.PersistentVolumeClaimSpec
func ParsePVCSpec(pvcspec string) (*corev1.PersistentVolumeClaimSpec, error) {
	pvcSpec := &corev1.PersistentVolumeClaimSpec{}
	err := yaml.NewYAMLOrJSONDecoder(strings.NewReader(pvcspec), len(pvcspec)).Decode(pvcSpec)
	if err != nil {
		return pvcSpec, fmt.Errorf("failed to decode the pvcspec string to PVC Spec: %v", err)
	}
	/*if err := json.Unmarshal([]byte(pvcspec), &pvcSpec); err != nil {
		return pvcSpec, fmt.Errorf("failed to unmarshal the pvcspec string to PVC Spec: %v", err)
	}*/
	return pvcSpec, nil
}
