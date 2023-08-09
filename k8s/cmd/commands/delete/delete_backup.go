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
package delete

import (
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/types"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deleteBackupLong = util.LongDesc(`
	Delete the backup job on kubernetes.`)

	deleteBackupExample = util.Examples(`
	# delete the default backup job
	vineyardctl delete backup`)
)

// deleteBackupCmd deletes the backup job on kubernetes
var deleteBackupCmd = &cobra.Command{
	Use:     "backup",
	Short:   "Delete the backup job on kubernetes",
	Long:    deleteBackupLong,
	Example: deleteBackupExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		log.Info("deleting Backup cr")
		backup := &v1alpha1.Backup{}
		if err := util.Delete(client, types.NamespacedName{
			Name:      flags.BackupName,
			Namespace: flags.GetDefaultVineyardNamespace(),
		}, backup); err != nil {
			log.Fatal(err, "failed to delete backup job")
		}
		log.Info("Backup Job is deleted.")
	},
}

func NewDeleteBackupCmd() *cobra.Command {
	return deleteBackupCmd
}

func init() {
	flags.ApplyBackupNameOpts(deleteBackupCmd)
}
