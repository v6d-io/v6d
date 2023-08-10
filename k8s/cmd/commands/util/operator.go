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
	"io/fs"
	"os"
	"path/filepath"

	"sigs.k8s.io/kustomize/kustomize/v4/commands/build"
	"sigs.k8s.io/kustomize/kyaml/filesys"

	"github.com/v6d-io/v6d/k8s/config"
)

func BuildKustomizeInEmbedDir() (Manifests, error) {
	fSys := filesys.MakeFsOnDisk()
	buffy := new(bytes.Buffer)
	cmd := build.NewCmdBuild(fSys, build.MakeHelp("", ""), buffy)

	// Create a temporary directory to extract the embedded config files
	tmpDir, err := os.MkdirTemp("", "v6d-operator-manifests-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	// Extract the embedded config files to the temporary directory
	if err := fs.WalkDir(config.Manifests, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			data, err := config.Manifests.ReadFile(path)
			if err != nil {
				return err
			}
			destPath := filepath.Join(tmpDir, path)
			if err := os.MkdirAll(filepath.Dir(destPath), os.ModePerm); err != nil {
				return err
			}
			if err := os.WriteFile(destPath, data, os.ModePerm); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}

	if err := cmd.RunE(cmd, []string{tmpDir + "/default"}); err != nil {
		return nil, err
	}
	return ParseManifestsToObjects(buffy.Bytes())
}
