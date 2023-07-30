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
)

func Test_getCertManagerManifestsFromLocal(t *testing.T) {
	tests := []struct {
		name    string
		want    Manifests
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name:    "ValidManifests",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := getCertManagerManifestsFromLocal()
			if (err != nil) != tt.wantErr {
				t.Errorf("getCertManagerManifestsFromLocal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}

func TestGetCertManager(t *testing.T) {
	tests := []struct {
		name    string
		want    Manifests
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name:    "ValidManifests",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := GetCertManager()
			if (err != nil) != tt.wantErr {
				t.Errorf("GetCertManager() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}
