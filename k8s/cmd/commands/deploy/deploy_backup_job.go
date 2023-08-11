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
	"k8s.io/apimachinery/pkg/api/resource"
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
	and pv spec and the related pv and pvc will be created automatically.
	Also, you could also specify the existing pv and pvc name to use`)

	deployBackupJobExample = util.Examples(`
	# deploy a backup job for all vineyard objects of the vineyard 
	# cluster on kubernetes and you could define the pv and pvc 
	# spec from json string as follows
	vineyardctl deploy backup-job \
		--vineyard-deployment-name vineyardd-sample \
		--vineyard-deployment-namespace vineyard-system  \
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
		--vineyard-deployment-name vineyardd-sample \
		--vineyard-deployment-namespace vineyard-system  \
		--path /var/vineyard/dump  \
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

	# deploy a backup job for specific vineyard objects of the vineyard 
	# cluster on kubernetes.
	cat pv-pvc.json | vineyardctl deploy backup-job \
		--vineyard-deployment-name vineyardd-sample \
		--vineyard-deployment-namespace vineyard-system  \
		--objectIDs "o000018d29207fd01,o000018d80d264010"  \
		--path /var/vineyard/dump
		
	# Assume you have already deployed a pvc named "pvc-sample", you 
	# could use them as the backend storage for the backup job as follows
	vineyardctl deploy backup-job \
		--vineyard-deployment-name vineyardd-sample \
		--vineyard-deployment-namespace vineyard-system  \
		--path /var/vineyard/dump  \
		--pvc-name pvc-sample
	
	# The namespace to deploy the backup and recover job must be the same
	# as the vineyard cluster namespace.
	# Assume the vineyard cluster is deployed in the namespace "test", you
	# could deploy the backup job as follows
	vineyardctl deploy backup-job \
		--vineyard-deployment-name vineyardd-sample \
		--vineyard-deployment-namespace test  \
		--namespace test  \
		--path /var/vineyard/dump  \
		--pvc-name pvc-sample`)
)

// deployBackupJobCmd deploy the backup job of vineyard cluster on kubernetes
var deployBackupJobCmd = &cobra.Command{
	Use:     "backup-job",
	Short:   "Deploy a backup job of vineyard cluster on kubernetes",
	Long:    deployBackupJobLong,
	Example: deployBackupJobExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgsOrInput(cmd, args)

		c := util.KubernetesClient()

		objs, err := getBackupObjectsFromTemplate(c, args)
		if err != nil {
			log.Fatal(err, "failed to get backup objects from template")
		}

		log.Info("applying backup manifests with owner reference")
		if err := util.ApplyManifestsWithOwnerRef(c, objs, "Job",
			"Role,Rolebinding"); err != nil {
			log.Fatal(err, "failed to apply backup objects")
		}

		log.Info("waiting backup job for ready")
		if err := waitBackupJobReady(c); err != nil {
			log.Fatal(err, "failed to wait backup job ready")
		}

		log.Info("Backup Job is ready.")
	},
}

func NewDeployBackupJobCmd() *cobra.Command {
	return deployBackupJobCmd
}

func init() {
	flags.ApplyDeployBackupJobOpts(deployBackupJobCmd)
}

func getBackupObjectsFromTemplate(c client.Client, args []string) ([]*unstructured.Unstructured, error) {
	backup, err := create.BuildBackup(c, args)
	if err != nil {
		return nil, err
	}

	// set the vineyardd name and namespace as the vineyard deployment
	backup.Spec.VineyarddName = flags.VineyardDeploymentName
	backup.Spec.VineyarddNamespace = flags.VineyardDeploymentNamespace
	pvcName := flags.PVCName
	if pvcName == "" {
		pvcName = flags.BackupName
	}
	opts := k8s.NewBackupOpts(
		flags.BackupName,
		pvcName,
		flags.BackupOpts.BackupPath,
	)
	backupCfg, err := opts.BuildCfgForVineyarctl(c, backup)
	if err != nil {
		return nil, err
	}

	tmplFunc := map[string]interface{}{
		"getResourceStorage": func(q resource.Quantity) string {
			return q.String()
		},
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
		jobName := flags.BackupName
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
