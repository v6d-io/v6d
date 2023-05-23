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

var (
	// RecoverName is the name of recover job
	RecoverName string

	// RecoverPath is the path of recover job
	RecoverPath string
)

func ApplyRecoverNameOpts(cmd *cobra.Command) {
	cmd.Flags().
		StringVarP(&RecoverName, "recover-name", "", "vineyard-recover",
			"the name of recover job")
}

func ApplyCreateRecoverOpts(cmd *cobra.Command) {
	ApplyRecoverNameOpts(cmd)
	cmd.Flags().
		StringVarP(&BackupName, "backup-name", "", "vineyard-backup",
			"the name of backup job")
}

func ApplyDeployRecoverJobOpts(cmd *cobra.Command) {
	ApplyRecoverNameOpts(cmd)
	cmd.Flags().
		StringVarP(&RecoverPath, "recover-path", "", "",
			"the path of recover job")
	cmd.Flags().
		StringVarP(&VineyardDeploymentName, "vineyard-deployment-name", "", "",
			"the name of vineyard deployment")
	cmd.Flags().
		StringVarP(&VineyardDeploymentNamespace, "vineyard-deployment-namespace", "", "",
			"the namespace of vineyard deployment")
	cmd.Flags().
		StringVarP(&PVCName, "pvc-name", "", "",
			"the name of an existing PersistentVolumeClaim")
}
