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
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestConvertToJson(t *testing.T) {
	// Test case 1: Valid JSON input
	jsonStr := `{"name":"John", "age": 31}`
	expectedJSON := `{"name":"John","age":31}`
	result, err := ConvertToJson(jsonStr)
	assert.NoError(t, err)
	assert.JSONEq(t, expectedJSON, result)

	// Test case 2: Valid YAML input
	yamlStr := `
name: John
age: 32
`
	expectedYAML := `{"name":"John","age":32}`
	result, err = ConvertToJson(yamlStr)
	assert.NoError(t, err)
	assert.JSONEq(t, expectedYAML, result)

}

func TestConvertToYaml(t *testing.T) {
	// Test case 1: Valid JSON input
	jsonStr := `{"name":"John","age":33}`
	expectedYAML := `age: 33
name: John
`
	result, err := ConvertToYaml(jsonStr)
	assert.NoError(t, err)
	assert.Equal(t, expectedYAML, result)

}

func TestParseEnvs(t *testing.T) {
	// Test case 1: Valid env array
	envArray := []string{"KEY1=Value1", "KEY2=Value2"}
	expectedEnvs := []corev1.EnvVar{
		{Name: "KEY1=Value1", Value: "KEY1=Value1"},
		{Name: "KEY2=Value2", Value: "KEY2=Value2"},
	}
	result, err := ParseEnvs(envArray)
	assert.NoError(t, err)
	assert.Equal(t, expectedEnvs, result)

	// Test case 2: Invalid env string without "=" separator
	invalidEnv := []string{"KEY1:Value1"}
	_, err = ParseEnvs(invalidEnv)
	assert.Error(t, err)
	assert.EqualError(t, err, "invalid env string")

	// Test case 3: Invalid env with empty name
	emptyNameEnv := []string{"=Value1"}
	_, err = ParseEnvs(emptyNameEnv)
	assert.Error(t, err)
	assert.EqualError(t, err, "the name of env can not be empty")

	// Test case 4: Invalid env with empty value
	emptyValueEnv := []string{"KEY1="}
	_, err = ParseEnvs(emptyValueEnv)
	assert.Error(t, err)
	assert.EqualError(t, err, "the value of env can not be empty")
}

func TestParsePVSpec(t *testing.T) {
	// Test case 1: Valid PV spec
	pvSpec := `{
		"capacity": {
			"storage": "1Gi"
		},
		"accessModes": ["ReadWriteOnce"],
		"persistentVolumeReclaimPolicy": "Retain"
	}`
	expectedSpec := corev1.PersistentVolumeSpec{
		Capacity: corev1.ResourceList{
			corev1.ResourceStorage: resource.MustParse("1Gi"),
		},
		AccessModes: []corev1.PersistentVolumeAccessMode{
			corev1.ReadWriteOnce,
		},
		PersistentVolumeReclaimPolicy: corev1.PersistentVolumeReclaimRetain,
	}
	result, err := ParsePVSpec(pvSpec)
	assert.NoError(t, err)
	assert.Equal(t, expectedSpec, *result)

	// Test case 2: Invalid PV spec
	invalidSpec := `{
		"capacity": {
			"storage": "1Gi"
		},
		"accessModes": "ReadWriteOnce", // Invalid accessModes
		"persistentVolumeReclaimPolicy": "Retain"
	}`
	_, err = ParsePVSpec(invalidSpec)
	assert.Error(t, err)
}

func TestParsePVCSpec(t *testing.T) {
	// Test case 1: Valid PVC spec
	pvcSpec := `{
		"accessModes": ["ReadWriteOnce"],
		"resources": {
			"requests": {
				"storage": "1Gi"
			}
		},
		"storageClassName": "standard_1"
	}`
	storageClassName := "standard_1"
	expectedSpec := corev1.PersistentVolumeClaimSpec{
		AccessModes: []corev1.PersistentVolumeAccessMode{
			corev1.ReadWriteOnce,
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceStorage: resource.MustParse("1Gi"),
			},
		},
		StorageClassName: &storageClassName,
	}
	result, err := ParsePVCSpec(pvcSpec)
	assert.NoError(t, err)
	assert.Equal(t, expectedSpec, *result)

	// Test case 2: Invalid PVC spec
	invalidSpec := `{
		"accessModes": "ReadWriteOnce", // Invalid accessModes
		"resources": {
			"requests": {
				"storage": "1Gi"
			}
		},
		"storageClassName": "standard_1"
	}`
	_, err = ParsePVCSpec(invalidSpec)
	assert.Error(t, err)
}

func TestParsePVandPVCSpec(t *testing.T) {
	// Test case 1: Valid JSON input
	jsonStr := `{
		"pv-spec": {
			"capacity": {
				"storage": "1Gi"
			},
			"accessModes": ["ReadWriteOnce"],
			"persistentVolumeReclaimPolicy": "Retain"
		},
		"pvc-spec": {
			"accessModes": ["ReadWriteOnce"],
			"resources": {
				"requests": {
					"storage": "1Gi"
				}
			},
			"storageClassName": "standard_2"
		}
	}`
	expectedPVSpec := corev1.PersistentVolumeSpec{
		Capacity: corev1.ResourceList{
			corev1.ResourceStorage: resource.MustParse("1Gi"),
		},
		AccessModes: []corev1.PersistentVolumeAccessMode{
			corev1.ReadWriteOnce,
		},
		PersistentVolumeReclaimPolicy: corev1.PersistentVolumeReclaimRetain,
	}
	storageClassName := "standard_2"
	expectedPVCSpec := corev1.PersistentVolumeClaimSpec{
		AccessModes: []corev1.PersistentVolumeAccessMode{
			corev1.ReadWriteOnce,
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceStorage: resource.MustParse("1Gi"),
			},
		},
		StorageClassName: &storageClassName,
	}
	pvSpec, pvcSpec, err := ParsePVandPVCSpec(jsonStr)
	assert.NoError(t, err)
	assert.Equal(t, expectedPVSpec, *pvSpec)
	assert.Equal(t, expectedPVCSpec, *pvcSpec)

}

func TestGetPVAndPVC(t *testing.T) {
	// Test case 1: Valid input
	pvAndPVC := `{
		"pv-spec": {
			"capacity": {
				"storage": "1Gi"
			},
			"accessModes": ["ReadWriteOnce"],
			"persistentVolumeReclaimPolicy": "Retain"
		},
		"pvc-spec": {
			"accessModes": ["ReadWriteOnce"],
			"resources": {
				"requests": {
					"storage": "1Gi"
				}
			},
			"storageClassName": "standard_3"
		}
	}`
	expectedPVSpec := corev1.PersistentVolumeSpec{
		Capacity: corev1.ResourceList{
			corev1.ResourceStorage: resource.MustParse("1Gi"),
		},
		AccessModes: []corev1.PersistentVolumeAccessMode{
			corev1.ReadWriteOnce,
		},
		PersistentVolumeReclaimPolicy: corev1.PersistentVolumeReclaimRetain,
	}
	storageClassName := "standard_3"
	expectedPVCSpec := corev1.PersistentVolumeClaimSpec{
		AccessModes: []corev1.PersistentVolumeAccessMode{
			corev1.ReadWriteOnce,
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceStorage: resource.MustParse("1Gi"),
			},
		},
		StorageClassName: &storageClassName,
	}
	pvSpec, pvcSpec, err := GetPVAndPVC(pvAndPVC)
	assert.NoError(t, err)
	assert.Equal(t, expectedPVSpec, *pvSpec)
	assert.Equal(t, expectedPVCSpec, *pvcSpec)

	// Test case 2: Empty input
	emptyInput := ""
	pvSpec, pvcSpec, err = GetPVAndPVC(emptyInput)
	assert.NoError(t, err)
	assert.Nil(t, pvSpec)
	assert.Nil(t, pvcSpec)

}

func TestParseOwnerRef(t *testing.T) {
	// Test case 1: Valid input
	ownerRefJSON := `[{
		"apiVersion": "v1",
		"kind": "Pod",
		"name": "my-pod",
		"uid": "12345",
		"controller": true,
		"blockOwnerDeletion": true
	}]`
	expectedOwnerRef := []metav1.OwnerReference{
		{
			APIVersion:         "v1",
			Kind:               "Pod",
			Name:               "my-pod",
			UID:                "12345",
			Controller:         boolPointer(true),
			BlockOwnerDeletion: boolPointer(true),
		},
	}
	result, err := ParseOwnerRef(ownerRefJSON)
	assert.NoError(t, err)
	assert.Equal(t, expectedOwnerRef, result)

	// Test case 2: Empty input
	emptyInput := ""
	result, err = ParseOwnerRef(emptyInput)
	assert.NoError(t, err)
	assert.Empty(t, result)

	// Test case 3: Invalid input
	invalidInput := `{"apiVersion": "v1", "kind": "Pod", "name": "my-pod"}`
	_, err = ParseOwnerRef(invalidInput)
	assert.Error(t, err)
}

// Helper function to return pointer to a boolean value
func boolPointer(b bool) *bool {
	return &b
}
