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
package flags

import "github.com/spf13/cobra"

const (
	defaultCertManagerVersion = "1.9.1"
)

var (
	// CertManagerVersion is the version of cert-manager
	CertManagerVersion string

	// WaitCertManager is the flag to indicate whether to wait for the cert-manager to be ready
	WaitCertManager bool
)

func ApplyCertManagerOpts(cmd *cobra.Command) {
	cmd.Flags().
		StringVarP(&CertManagerVersion, "version", "v", defaultCertManagerVersion,
			"the version of cert-manager")
	// Always wait for the cert-manager to be ready by default
	// to avoid errors caused by the cert-manager not ready
	cmd.Flags().
		BoolVarP(&WaitCertManager, "wait", "", true, "wait for the cert-manager to be ready")
}
