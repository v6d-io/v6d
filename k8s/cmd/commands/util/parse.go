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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
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

func ParsePVandPVCSpec(PvAndPvc string) (*corev1.PersistentVolumeSpec, *corev1.PersistentVolumeClaimSpec, error) {
	var result map[string]interface{}
	err := json.Unmarshal([]byte(PvAndPvc), &result)
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

// ParsePVSpec parse the json string to corev1.PersistentVolumeSpec
func ParsePVSpec(pvspec string) (*corev1.PersistentVolumeSpec, error) {
	// add the spec field to the pvspec string
	pvspec = `{"spec":` + pvspec + `}`

	decoder := Deserializer()
	obj, _, err := decoder.Decode([]byte(pvspec), &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolume"}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode the pvspec string to PV Spec: %v", err)
	}

	pv := obj.(*corev1.PersistentVolume)
	return &pv.Spec, nil
}

// ParsePVCSpec parse the json string to corev1.PersistentVolumeClaimSpec
func ParsePVCSpec(pvcspec string) (*corev1.PersistentVolumeClaimSpec, error) {
	// add the spec field to the pvcspec string
	pvcspec = `{"spec":` + pvcspec + `}`

	decoder := Deserializer()
	obj, _, err := decoder.Decode([]byte(pvcspec), &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolumeClaim"}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode the pvspec string to PV Spec: %v", err)
	}

	pvc := obj.(*corev1.PersistentVolumeClaim)
	return &pvc.Spec, nil
}
