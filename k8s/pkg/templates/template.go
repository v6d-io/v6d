/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

// Package templates contains all template related functions
package templates

import (
	"embed"
	"path/filepath"

	"github.com/pkg/errors"
)

//go:embed vineyardd etcd operation sidecar backup recover csidriver
var fs embed.FS

// ReadFile reads a file from the embed.FS
func ReadFile(path string) ([]byte, error) {
	return fs.ReadFile(path)
}

// GetFilesRecursive returns all files in a directory
func GetFilesRecursive(path string) ([]string, error) {
	dir := filepath.Join(filepath.Dir(path), path)
	fd, err := fs.ReadDir(dir)
	if err != nil {
		return []string{}, errors.Wrap(err, "ReadDir error")
	}
	files := []string{}
	for _, f := range fd {
		if f.IsDir() {
			subFiles, err := GetFilesRecursive(filepath.Join(dir, f.Name()))
			if err != nil {
				return []string{}, errors.Wrap(err, "GetFilesRecursive error")
			}
			files = append(files, subFiles...)
		} else {
			files = append(files, filepath.Join(dir, f.Name()))
		}
	}
	return files, nil
}

// Implement the `Repo` interface "skywalking-swck/operator/pkg/kubernetes"
type embedRepo struct{}

func (*embedRepo) ReadFile(path string) ([]byte, error) {
	return ReadFile(path)
}

func (*embedRepo) GetFilesRecursive(path string) ([]string, error) {
	return GetFilesRecursive(path)
}

var Repo = &embedRepo{}
