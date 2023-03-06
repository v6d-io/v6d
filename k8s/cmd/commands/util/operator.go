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
	"bytes"
	"fmt"

	"sigs.k8s.io/kustomize/kustomize/v4/commands/build"
	"sigs.k8s.io/kustomize/kyaml/filesys"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

// default kustomize dir from github repo
var defaultKustomizeDir = "https://github.com/v6d-io/v6d/k8s/config/default?submodules=false"

func GetKustomizeDir() string {
	if flags.KustomizeDir != "" {
		return flags.KustomizeDir
	}
	if flags.OperatorVersion != "dev" {
		return fmt.Sprintf("%s&ref=v%s", defaultKustomizeDir, flags.OperatorVersion)
	}
	return defaultKustomizeDir
}

func BuildKustomizeInDir(kustomizeDir string) (Manifests, error) {
	fSys := filesys.MakeFsOnDisk()
	buffy := new(bytes.Buffer)
	cmd := build.NewCmdBuild(fSys, build.MakeHelp("", ""), buffy)

	if err := cmd.RunE(cmd, []string{kustomizeDir}); err != nil {
		return nil, err
	}
	return ParseManifestsToObjects(buffy.Bytes())
}
