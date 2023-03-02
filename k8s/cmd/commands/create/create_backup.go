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

var (
	createBackupLong = util.LongDesc(`
	Backup the current vineyard cluster on kubernetes. You could backup all objects 
	of the current vineyard cluster quickly. For persistent storage, you could specify 
	the pv spec and pv spec.`)

	createBackupExample = util.Examples(`
	# create a backup job for the vineyard cluster on kubernetes 
	# you could define the pv and pvc spec from json string as follows
	vineyardctl create backup \
		--vineyardd-name vineyardd-sample \
		--vineyardd-namespace vineyard-system  \
		--limit 1000 \
		--path /var/vineyard/dump  \
		--pv-pvc-spec '{
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
	# you could define the pv and pvc spec from yaml string as follows
	vineyardctl create backup \
		--vineyardd-name vineyardd-sample \
		--vineyardd-namespace vineyard-system  \
		--limit 1000 --path /var/vineyard/dump  \
		--pv-pvc-spec  \
		'
		pv-spec:
		capacity:
			storage: 1Gi
		accessModes:
		- ReadWriteOnce
		storageClassName: manual
		hostPath:
			path: "/var/vineyard/dump"
		pvc-spec:
		storageClassName: manual
		accessModes:
		- ReadWriteOnce
		resources:
			requests:
			storage: 1Gi
		'
	
	# create a backup job for the vineyard cluster on kubernetes
	# you could define the pv and pvc spec from json file as follows
	# also you could use yaml file instead of json file
	cat pv-pvc.json | vineyardctl create backup \
		--vineyardd-name vineyardd-sample \
		--vineyardd-namespace vineyard-system  \
		--limit 1000 --path /var/vineyard/dump  \
		-`)
)

// createBackupCmd creates the backup job of vineyard cluster on kubernetes
var createBackupCmd = &cobra.Command{
	Use:     "backup",
	Short:   "Backup the current vineyard cluster on kubernetes",
	Long:    createBackupLong,
	Example: createBackupExample,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) > 0 && args[0] != "-" {
			util.ErrLogger.Fatal("invalid argument: ", args)
		}

		// Check if the input is coming from stdin
		str, err := util.ReadJsonFromStdin(args)
		if err != nil {
			util.ErrLogger.Fatalf("failed to parse from stdin: %v", err)
		}
		if str != "" {
			flags.BackupPVandPVC = str
		}
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
	opts := &flags.BackupOpts

	if backupPVandPVC != "" {
		backupPVandPVCJson, err := util.ConvertToJson(backupPVandPVC)
		if err != nil {
			return nil, errors.Wrap(err, "failed to convert the pv and pvc of backup to json")
		}
		backupPVSpec, backupPVCSpec, err := util.ParsePVandPVCSpec(backupPVandPVCJson)
		if err != nil {
			return nil, errors.Wrap(err, "failed to parse the pv and pvc of backup")
		}
		opts.PersistentVolumeSpec = *backupPVSpec
		opts.PersistentVolumeClaimSpec = *backupPVCSpec
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
