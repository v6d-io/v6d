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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

func TestGetKustomizeDir(t *testing.T) {
	// Save the original values so we can restore them later.
	origKustomizeDir := flags.KustomizeDir
	origOperatorVersion := flags.OperatorVersion
	defer func() {
		flags.KustomizeDir = origKustomizeDir
		flags.OperatorVersion = origOperatorVersion
	}()

	// Case 1: flags.KustomizeDir is set.
	flags.KustomizeDir = "https://github.com/v6d-io/v6d/k8s/config/test"
	flags.OperatorVersion = "dev"
	assert.Equal(t, flags.KustomizeDir, GetKustomizeDir())

	// Case 2: flags.KustomizeDir is not set and OperatorVersion is not "dev".
	flags.KustomizeDir = ""
	flags.OperatorVersion = "1.0.0"
	expected := fmt.Sprintf("%s&ref=v%s", defaultKustomizeDir, flags.OperatorVersion)
	assert.Equal(t, expected, GetKustomizeDir())

	// Case 3: flags.KustomizeDir is not set and OperatorVersion is "dev".
	flags.KustomizeDir = ""
	flags.OperatorVersion = "dev"
	assert.Equal(t, defaultKustomizeDir, GetKustomizeDir())
}

func TestBuildKustomizeInDir(t *testing.T) {
	// Test valid directory
	kustomizeDir := "/home/zhuyi/v6d/k8s/config/certmanager/" // replace this with a valid Kustomize directory path
	_, err := BuildKustomizeInDir(kustomizeDir)
	assert.Nil(t, err)

	// Test invalid directory
	kustomizeDir = "./path/to/invalid/kustomize/dir" // replace this with an invalid Kustomize directory path
	_, err = BuildKustomizeInDir(kustomizeDir)
	assert.NotNil(t, err)
}
