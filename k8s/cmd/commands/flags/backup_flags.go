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

import (
	"github.com/spf13/cobra"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
)

var (
	// BackupName is the name of backup job
	BackupName string

	// BackupOpts holds all configuration of backup Spec
	BackupOpts v1alpha1.BackupSpec

	// BackupPVandPVC is the string of PersistentVolume data and
	// PersistentVolumeClaim data
	BackupPVandPVC string
)

func ApplyBackupNameOpts(cmd *cobra.Command) {
	cmd.Flags().
		StringVarP(&BackupName, "backup-name", "", "vineyard-backup",
			"the name of backup job")
}

func ApplyBackupOpts(cmd *cobra.Command) {
	// the following flags are used to build the backup configurations
	cmd.Flags().
		StringVarP(&BackupOpts.VineyarddName, "vineyardd-name", "", "",
			"the name of vineyardd")
	cmd.Flags().
		StringVarP(&BackupOpts.VineyarddNamespace, "vineyardd-namespace", "", "",
			"the namespace of vineyardd")
	cmd.Flags().IntVarP(&BackupOpts.Limit, "limit", "", 1000,
		"the limit of objects to backup")
	cmd.Flags().StringVarP(&BackupOpts.BackupPath, "path", "", "",
		"the path of the backup data")
	cmd.Flags().StringVarP(&BackupPVandPVC, "pv-pvc-spec", "", "",
		"the PersistentVolume and PersistentVolumeClaim of the backup data")

	// the following flags are used to build the backup job
	ApplyBackupNameOpts(cmd)
}
