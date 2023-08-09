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
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/pkg/errors"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/create"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	deployRecoverJobLong = util.LongDesc(`
	Deploy the recover job for vineyard cluster on kubernetes, which
	will recover all objects from a backup of vineyard cluster. Usually,
	the recover job should be created in the same namespace of
	the backup job.`)

	deployRecoverJobExample = util.Examples(`
	# Deploy a recover job for the vineyard deployment in the same namespace.
	# After the recover job finished, the command will create a kubernetes
	# configmap named [recover-name]+"-mapping-table" that contains the
	# mapping table from the old vineyard objects to the new ones.
	#
	# If you create the recover job as follows, you can get the mapping table via
	# "kubectl get configmap vineyard-recover-mapping-table -n vineyard-system -o yaml"
	# the left column is the old object id, and the right column is the new object id.
	vineyardctl deploy recover-job \
	--vineyard-deployment-name vineyardd-sample \
	--vineyard-deployment-namespace vineyard-system  \
	--recover-path /var/vineyard/dump \
	--pvc-name vineyard-backup`)
)

// deployRecoverJobCmd creates the recover job of vineyard cluster on kubernetes
var deployRecoverJobCmd = &cobra.Command{
	Use:     "recover-job",
	Short:   "Deploy a recover job to recover a backup of current vineyard cluster on kubernetes",
	Long:    deployRecoverJobLong,
	Example: deployRecoverJobExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)
		client := util.KubernetesClient()
		clientset := util.KubernetesClientset()
		util.CreateNamespaceIfNotExist(client)

		objs, err := getRecoverObjectsFromTemplate(client)
		if err != nil {
			log.Fatal(err, "failed to get recover objects from template")
		}

		log.Info("applying recover objects with owner ref")
		if err := util.ApplyManifestsWithOwnerRef(client, objs, "Job", "Role,Rolebinding"); err != nil {
			log.Fatal(err, "failed to apply recover objects")
		}

		log.Info("waiting recover job for ready")
		if err := waitRecoverJobReady(client); err != nil {
			log.Fatal(err, "failed to wait recover job ready")
		}

		if err := createMappingTableConfigmap(client, *clientset); err != nil {
			log.Fatal(err, "failed to create mapping table configmap")
		}
		log.Info("Recover job is ready.")
	},
}

func NewDeployRecoverJobCmd() *cobra.Command {
	return deployRecoverJobCmd
}

func init() {
	flags.ApplyDeployRecoverJobOpts(deployRecoverJobCmd)
}

func getRecoverObjectsFromTemplate(c client.Client) ([]*unstructured.Unstructured, error) {
	recover, err := create.BuildV1alphaRecoverCR()
	if err != nil {
		log.Fatal(err, "failed to build recover cr")
	}

	dummyBackup := v1alpha1.Backup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.BackupName,
			Namespace: flags.Namespace,
		},
		Spec: v1alpha1.BackupSpec{
			VineyarddName:      flags.VineyardDeploymentName,
			VineyarddNamespace: flags.VineyardDeploymentNamespace,
			BackupPath:         flags.RecoverPath,
		},
	}

	opts := k8s.NewRecoverOpts(
		flags.RecoverName,
		flags.PVCName,
		flags.RecoverPath,
	)
	recoverCfg, err := opts.BuildCfgForVineyarctl(c, &dummyBackup)
	if err != nil {
		return nil, err
	}
	tmplFunc := map[string]interface{}{
		"getRecoverConfig": func() k8s.RecoverConfig {
			return recoverCfg
		},
	}

	objects, err := util.BuildObjsFromManifests("recover", recover, tmplFunc)
	if err != nil {
		return nil, err
	}

	return objects, nil
}

func waitRecoverJobReady(c client.Client) error {
	return util.Wait(func() (bool, error) {
		jobName := flags.RecoverName
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

func createMappingTableConfigmap(c client.Client, cs kubernetes.Clientset) error {
	jobName := flags.RecoverName
	jobNamespace := flags.Namespace
	mappingTable, err := k8s.GetObjectMappingTable(c, cs, jobName, jobNamespace)
	if err != nil {
		return errors.Wrap(err, "failed to get object mapping table")
	}

	cm := corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ConfigMap",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName + "-mapping-table",
			Namespace: jobNamespace,
		},
		Data: mappingTable,
	}
	err = c.Create(context.Background(), &cm)
	if apierrors.IsAlreadyExists(err) {
		err := c.Update(context.Background(), &cm)
		if err != nil {
			return errors.Wrap(err, "failed to update mapping table configmap")
		}
	}
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrap(err, "failed to create mapping table configmap")
	}
	return nil
}
