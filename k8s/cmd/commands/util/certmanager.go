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
	"io"
	"net/http"
	"strings"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

// GetCertManagerURL get the url of cert-manager
func GetCertManagerURL() string {
	if flags.CertManagerVersion != "1.9.1" {
		return strings.Replace(flags.DefaultCertManagerManifestURL, "v1.9.1", "v"+flags.CertManagerVersion, 1)
	}
	return flags.DefaultCertManagerManifestURL
}

// GetCertManagerManifests get the manifests from the url
func GetCertManagerManifests(certManagerManifestURL string) (string, error) {
	resp, err := http.Get(certManagerManifestURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	manifests, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(manifests), nil
}
