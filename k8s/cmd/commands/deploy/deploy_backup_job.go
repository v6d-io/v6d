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
package deploy

import (
	"context"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	batchv1 "k8s.io/api/batch/v1"

	"github.com/v6d-io/v6d/k8s/cmd/commands/create"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployBackupJobLong = util.LongDesc(`
	Deploy the backup job for the vineyard cluster on kubernetes,
	which will backup all objects of the current vineyard cluster
	quickly. For persistent storage, you could specify the pv spec
	and pv spec.`)

	deployBackupJobExample = util.Examples(`
	# deploy a backup job for the vineyard cluster on kubernetes
	# you could define the pv and pvc spec from json string as follows
	vineyardctl deploy backup-job \
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

	# deploy a backup job for the vineyard cluster on kubernetes
	# you could define the pv and pvc spec from yaml string as follows
	vineyardctl deploy backup-job \
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

	# deploy a backup job for the vineyard cluster on kubernetes
	# you could define the pv and pvc spec from json file as follows
	# also you could use yaml file instead of json file
	cat pv-pvc.json | vineyardctl deploy backup-job \
		--vineyardd-name vineyardd-sample \
		--vineyardd-namespace vineyard-system  \
		--limit 1000 --path /var/vineyard/dump  \
		-
	`)
)

// deployBackupJobCmd deploy the backup job of vineyard cluster on kubernetes
var deployBackupJobCmd = &cobra.Command{
	Use:     "backup-job",
	Short:   "Deploy a backup job of vineyard cluster on kubernetes",
	Long:    deployBackupJobLong,
	Example: deployBackupJobExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgsOrInput(cmd, args)

		client := util.KubernetesClient()

		objs, err := getBackupObjectsFromTemplate(client, args)
		if err != nil {
			log.Fatal(err, "failed to get backup objects from template")
		}
		for _, obj := range objs {
			if err := util.CreateIfNotExists(client, obj); err != nil {
				log.Fatal(err, "failed to create backup objects")
			}
		}

		waitBackupJobReady(client)

		log.Info("Backup Job is ready.")
	},
}

func NewDeployBackupJobCmd() *cobra.Command {
	return deployBackupJobCmd
}

func init() {
	flags.ApplyBackupOpts(deployBackupJobCmd)
}

func getBackupObjectsFromTemplate(c client.Client, args []string) ([]*unstructured.Unstructured, error) {
	backup, err := create.BuildBackup(c, args)
	if err != nil {
		return nil, err
	}

	useVineyardScheduler := false
	path := flags.BackupOpts.BackupPath
	backupCfg, err := k8s.BuildBackupCfg(c, flags.BackupName, backup, path, useVineyardScheduler)
	if err != nil {
		return nil, err
	}
	tmplFunc := map[string]interface{}{
		"getBackupConfig": func() k8s.BackupConfig {
			return backupCfg
		},
	}

	objects, err := util.BuildObjsFromManifests("backup", backup, tmplFunc)
	if err != nil {
		return nil, err
	}

	return objects, nil
}

func waitBackupJobReady(c client.Client) error {
	return util.Wait(func() (bool, error) {
		jobName := "backup-" + flags.VineyarddName + "-" + flags.VineyarddNamespace
		name := client.ObjectKey{Name: jobName, Namespace: flags.Namespace}
		job := batchv1.Job{}
		if err := c.Get(context.TODO(), name, &job); err != nil {
			return false, err
		}
		if job.Status.Succeeded == *job.Spec.Parallelism {
			return true, nil
		}
		return false, nil
	})
}
