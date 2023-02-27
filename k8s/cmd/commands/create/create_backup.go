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
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
)

// createBackupCmd creates the backup job of vineyard cluster on kubernetes
var createBackupCmd = &cobra.Command{
	Use:   "backup",
	Short: "Backup the current vineyard cluster on kubernetes",
	Long: `Backup the current vineyard cluster on kubernetes. You could backup all objects of
the current vineyard cluster quickly. For persistent storage, you could specify the pv spec and
pv spec.

For example:

# the json format of pv spec and pvc spec is as follows:
go run cmd/main.go create backup --pv-pvc-spec '{
"pv-spec": {
	"capacity": {
	  "storage": "1Gi"
	},
	"accessModes": [
	  "ReadWriteOnce"
	],
	"storageClassName": "manual",
	"hostPath": {
	  "path": "/var/vineyard/dump"
	}
},
"pvc-spec": {
	"storageClassName": "manual",
	"accessModes": [
	  "ReadWriteOnce"
	],
	"resources": {
	  "requests": {
		"storage": "1Gi"
	  }
	}
}
}'

# create a backup job for the vineyard cluster on kubernetes
vineyardctl create backup \
--vineyardd-name vineyardd-sample \
--vineyardd-namespace vineyard-system  \
--limit 1000 --path /var/vineyard/dump  \
--pv-spec \
'{
	"capacity": {
	  "storage": "1Gi"
	},
	"accessModes": [
	  "ReadWriteOnce"
	],
	"storageClassName": "manual",
	"hostPath": {
	  "path": "/var/vineyard/dump"
	}
}' \
--pvc-spec \
'{
	"storageClassName": "manual",
	"accessModes": [
	  "ReadWriteOnce"
	],
	"resources": {
	  "requests": {
		"storage": "1Gi"
	  }
	}
}'`,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()

		backup, err := buildBackupJob()
		if err != nil {
			util.ErrLogger.Fatalf("failed to build backup job: %+v", err)
		}

		if err := util.Create(client, backup, func(backup *v1alpha1.Backup) bool {
			return backup.Status.State != k8s.SucceedState
		}); err != nil {
			util.ErrLogger.Fatalf("failed to create/wait backup job: %+v", err)
		}
		util.InfoLogger.Println("Backup Job is ready.")
	},
}

func NewCreateBackupCmd() *cobra.Command {
	return createBackupCmd
}

func init() {
	flags.ApplyBackupOpts(createBackupCmd)
}

func buildBackupJob() (*v1alpha1.Backup, error) {
	backupPVandPVC := flags.BackupPVandPVC
	// backupPV := flags.BackupPVSpec
	// backupPVC := flags.BackupPVCSpec
	opts := &flags.BackupOpts

	if backupPVandPVC != "" {
		backupPVSpec, backupPVCSpec, err := util.ParsePVandPVCSpec(backupPVandPVC)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pv and pvc of backup")
		}
		opts.PersistentVolumeSpec = *backupPVSpec
		opts.PersistentVolumeClaimSpec = *backupPVCSpec
	}
	/*if backupPV != "" {
		backupPV, err := util.ParsePVSpec(backupPV)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pv of backup")
		}
		opts.PersistentVolumeSpec = *backupPV
	}

	if backupPVC != "" {
		backupPVC, err := util.ParsePVCSpec(backupPVC)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pvc of backup")
		}
		opts.PersistentVolumeClaimSpec = *backupPVC
	}*/

	backup := &v1alpha1.Backup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.BackupName,
			Namespace: flags.Namespace,
		},
		Spec: *opts,
	}
	return backup, nil
}
