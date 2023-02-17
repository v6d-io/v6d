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
package create

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// createBackupCmd creates the backup job of vineyard cluster on kubernetes
var createBackupCmd = &cobra.Command{
	Use:   "backup",
	Short: "Backup the current vineyard cluster on kubernetes",
	Long: `Backup the current vineyard cluster on kubernetes. You could backup all objects of 
the current vineyard cluster quickly. For persistent storage, you could specify the pv spec and
pv spec.

For example:

# create a backup job for the vineyard cluster on kubernetes
vineyardctl create backup --vineyardd-name vineyardd-sample --vineyardd-namespace vineyard-system  \
--limit 1000 --path /var/vineyard/dump  \
--pv-spec '{"capacity": {"storage":"1Gi"}, "accessModes": ["ReadWriteOnce"], "storageClassName": "manual", "hostPath": {"path": "/var/vineyard/dump"}}'  \
--pvc-spec '{"storageClassName": "manual", "accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "1Gi"}}}'`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := util.ValidateNoArgs("create backup", args); err != nil {
			log.Fatal("failed to validate create backup command args and flags: ", err)
		}
		scheme, err := util.GetOperatorScheme()
		if err != nil {
			log.Fatal("failed to get operator scheme: ", err)
		}

		kubeClient, err := util.GetKubeClient(scheme)
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		backup, err := buildBackupJob()
		if err != nil {
			log.Fatal("failed to build backup job: ", err)
		}

		if err := kubeClient.Create(context.TODO(), backup); err != nil {
			log.Fatal("failed to create backup job: ", err)
		}

		if err := waitBackupJobDone(kubeClient, backup); err != nil {
			log.Fatal("failed to wait backup job done: ", err)
		}
		log.Println("Backup Job is ready.")
	},
}

func buildBackupJob() (*v1alpha1.Backup, error) {
	backupPV := flags.BackupPVSpec
	backupPVC := flags.BackupPVCSpec
	opts := &flags.BackupOpts
	if backupPV != "" {
		backupPV, err := util.ParsePVSpec(backupPV)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the pv of backup: %v", err)
		}
		opts.PersistentVolumeSpec = *backupPV
	}

	if backupPVC != "" {
		backupPVC, err := util.ParsePVCSpec(backupPVC)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the pvc of backup: %v", err)
		}
		opts.PersistentVolumeClaimSpec = *backupPVC
	}

	backup := &v1alpha1.Backup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.BackupName,
			Namespace: flags.Namespace,
		},
		Spec: *opts,
	}
	return backup, nil
}

// wait for the backup job to be done
func waitBackupJobDone(c client.Client, backup *v1alpha1.Backup) error {
	return wait.PollImmediate(1*time.Second, 600*time.Second, func() (bool, error) {
		err := c.Get(context.TODO(), client.ObjectKey{
			Name:      backup.Name,
			Namespace: backup.Namespace,
		}, backup)
		if err != nil || backup.Status.State != k8s.SucceedState {
			return false, nil
		}
		return true, nil
	})
}

func NewCreateBackupCmd() *cobra.Command {
	return createBackupCmd
}

func init() {
	flags.NewBackupOpts(createBackupCmd)
}
