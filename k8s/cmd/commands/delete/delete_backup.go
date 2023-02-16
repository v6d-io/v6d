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
	"context"
	"log"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
)

// deleteBackupCmd deletes the backup job on kubernetes
var deleteBackupCmd = &cobra.Command{
	Use:   "backup",
	Short: "Delete the backup job on kubernetes",
	Long: `Delete the backup job on kubernetes. 
For example:

# delete the default backup job
vineyardctl delete backup`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("delete backup", args); err != nil {
			log.Fatal("failed to validate delete backup args and flags: ", err)
		}

		kubeClient, err := util.GetKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		backup := &v1alpha1.Backup{}
		if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: flags.BackupName,
			Namespace: flags.GetDefaultVineyardNamespace()},
			backup); err != nil && !apierrors.IsNotFound(err) {
			log.Fatal("failed to get backup job: ", err)
		}

		if err := kubeClient.Delete(context.Background(), backup); err != nil {
			log.Fatal("failed to delete backup job: ", err)
		}

		log.Println("Backup Job is deleted.")
	},
}

func NewDeleteBackupCmd() *cobra.Command {
	return deleteBackupCmd
}

func init() {
	flags.NewBackupNameOpts(deleteBackupCmd)
}
