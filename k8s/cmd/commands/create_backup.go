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
package commands

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/spf13/cobra"
	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// createBackupCmd creates the backup of vineyard cluster on kubernetes
var createBackupCmd = &cobra.Command{
	Use:   "backup",
	Short: "Backup the current vineyard cluster on kubernetes",
	Long: `Backup the current vineyard cluster on kubernetes. You could backup all objects of 
the current vineyard cluster quickly. For persistent backup, you could specify the pv spec and
pv spec.

For example:

# create a backup job for the vineyard cluster on kubernetes
vineyardctl create backup --vineyardd-name vineyardd-sample --vineyardd-namespace vineyard-system  \
--limit 1000 --path /var/vineyard/dump  \
--pv-spec '{"Capacity": "1Gi", "AccessModes": ["ReadWriteOnce"], "StorageClassName": "Manual", "HostPath": {"Path": "/var/vineyard/dump"}}'  \
--pvc-spec '{"storageClassName": "manual", "accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "1Gi"}}}'`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := ValidateNoArgs("create backup", args); err != nil {
			log.Fatal("failed to validate create backup command args and flags: ", err)
		}

		kubeClient, err := getKubeClient()
		if err != nil {
			log.Fatal("failed to get kubeclient: ", err)
		}

		backup, err := buildBackupJob()
		if err != nil {
			log.Fatal("failed to build backup job: ", err)
		}

		if err := waitBackupJobDone(kubeClient, backup); err != nil {
			log.Fatal("failed to wait backup job done: ", err)
		}
		log.Println("Backup Job is ready.")
	},
}

func buildBackupJob() (*v1alpha1.Backup, error) {
	if BackupPVSpec != "" {
		backupPV, err := ParsePVSpec(BackupPVSpec)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the pv of backup: %v", err)
		}
		BackupOpts.PersistentVolumeSpec = *backupPV
	}

	if BackupPVCSpec != "" {
		backupPVC, err := ParsePVCSpec(BackupPVCSpec)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the pvc of backup: %v", err)
		}
		BackupOpts.PersistentVolumeClaimSpec = *backupPVC
	}

	backup := &v1alpha1.Backup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BackupName,
			Namespace: GetDefaultVineyardNamespace(),
		},
		Spec: BackupOpts,
	}
	return backup, nil
}

// wait for the backup job to be done
func waitBackupJobDone(c client.Client, backup *v1alpha1.Backup) error {
	return wait.PollImmediate(1*time.Second, 300*time.Second, func() (bool, error) {
		err := c.Create(context.TODO(), backup)
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return false, nil
		}
		return true, nil
	})
}

// BackupName is the name of backup
var BackupName string

// BackupPVSpec is the PersistentVolumeSpec of the backup data
var BackupPVSpec string

// BackupPVCSpec is the PersistentVolumeClaimSpec of the backup data
var BackupPVCSpec string

// BackupOpts holds all configuration of backup Spec
var BackupOpts v1alpha1.BackupSpec

func NewCreateBackupCmd() *cobra.Command {
	return createBackupCmd
}

func init() {
	// the following flags are used to build the backup configurations
	createBackupCmd.Flags().StringVarP(&BackupOpts.VineyarddName, "vineyardd-name", "", "", "the name of vineyardd")
	createBackupCmd.Flags().StringVarP(&BackupOpts.VineyarddNamespace, "vineyardd-namespace", "", "", "the namespace of vineyardd")
	createBackupCmd.Flags().IntVarP(&BackupOpts.Limit, "limit", "", 1000, "the limit of objects to backup")
	createBackupCmd.Flags().StringVarP(&BackupOpts.BackupPath, "path", "", "", "the path of the backup data")
	createBackupCmd.Flags().StringVarP(&BackupPVSpec, "pv-spec", "", "", "the PersistentVolumeSpec of the backup data")
	createBackupCmd.Flags().StringVarP(&BackupPVCSpec, "pvc-spec", "", "", "the PersistentVolumeClaimSpec of the backup data")

	// the following flags are used to build the backup job
	createBackupCmd.Flags().StringVarP(&BackupName, "backup-name", "", "vineyard-backup", "the name of backup job")
}
