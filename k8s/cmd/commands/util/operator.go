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
	"embed"
	"fmt"
	"io/fs"
	"log"

	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/krusty"

	"github.com/v6d-io/v6d/k8s/config"
)

func writeEmbeddedFileToKustomizeFS(efs embed.FS, kfs filesys.FileSystem, path string) error {
	data, err := efs.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read path %s from embedded file system: %w", path, err)
	}

	if err := kfs.WriteFile(path, data); err != nil {
		return fmt.Errorf("failed to write path %s to in-memory kustomize file system: %w", path, err)
	}
	return nil
}

// ConvertEmbeddedFSToKustomizeFS performs a walk from the root of the provided
// embed.FS and copies the file tree into an in-memory Kustomize FileSystem,
func ConvertEmbeddedFSToKustomizeFS(efs embed.FS) (filesys.FileSystem, error) {
	fsys := filesys.MakeFsInMemory()
	err := fs.WalkDir(config.Manifests, ".", func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return fmt.Errorf("encountered a walk error: %w", walkErr)
		}

		if d.IsDir() {
			return fsys.Mkdir(path)
		}

		return writeEmbeddedFileToKustomizeFS(efs, fsys, path)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to walk embedded file system: %w", err)
	}

	return fsys, nil
}

func BuildKustomizeInEmbedDir() (Manifests, error) {
	fSys, err := ConvertEmbeddedFSToKustomizeFS(config.Manifests)
	if err != nil {
		log.Fatalf("failed to convert embedded file system to kustomize file system: %v", err)
	}
	k := krusty.MakeKustomizer(fSys, krusty.MakeDefaultOptions())

	resMap, err := k.Run("default")
	if err != nil {
		log.Fatalf("failed to run kustomize build: %v", err)
	}

	yamls, err := resMap.AsYaml()
	if err != nil {
		log.Fatalf("failed to convert resources to YAML: %v", err)
	}

	return ParseManifestsToObjects(yamls)
}
