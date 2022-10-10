/** Copyright 2020-2022 Alibaba Group Holding Limited.

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
	"fmt"
	"path/filepath"
)

//go:embed vineyardd etcd assembly
var fs embed.FS

// EmbedTemplate is only used for implementing the interface
type EmbedTemplate struct{}

// NewEmbedTemplate returns a new EmbedTemplate
func NewEmbedTemplate() *EmbedTemplate {
	return &EmbedTemplate{}
}

// ReadFile reads a file from the embed.FS
func (e *EmbedTemplate) ReadFile(path string) ([]byte, error) {
	return fs.ReadFile(path)
}

// GetFilesRecursive returns all files in a directory
func (e *EmbedTemplate) GetFilesRecursive(dir string) ([]string, error) {
	path := filepath.Join(filepath.Dir(dir), dir)
	fd, err := fs.ReadDir(path)
	if err != nil {
		return []string{}, fmt.Errorf("ReadDir Error: %v", err)
	}
	files := []string{}
	for _, f := range fd {
		if !f.IsDir() {
			files = append(files, filepath.Join(path, f.Name()))
		}
	}
	return files, nil
}
