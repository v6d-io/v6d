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
	"time"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	"github.com/v6d-io/v6d/k8s/controllers/k8s"
)

// createRecoverCmd creates the recover job of vineyard cluster on kubernetes
var createRecoverCmd = &cobra.Command{
	Use:   "recover",
	Short: "Recover the current vineyard cluster on kubernetes",
	Long: `Recover the current vineyard cluster on kubernetes. You could recover all objects from
a backup of vineyard cluster. Usually, the recover job should be created in the same namespace of
the backup job.

For example:

# create a recover job for a backup job in the same namespace
vineyardctl create recover --backup-name vineyardd-sample -n vineyard-system`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := cobra.NoArgs(cmd, args); err != nil {
			util.ErrLogger.Fatal(err)
		}
		client := util.KubernetesClient()

		recover, err := buildRecoverJob()
		if err != nil {
			util.ErrLogger.Fatal("failed to build recover job: ", err)
		}

		if err := client.Create(context.TODO(), recover); err != nil {
			util.ErrLogger.Fatal("failed to create recover job: ", err)
		}

		if err := waitRecoverJobDone(client, recover); err != nil {
			util.ErrLogger.Fatal("failed to wait backup job done: ", err)
		}
		util.InfoLogger.Println("Backup Job is ready.")
	},
}

func NewCreateRecoverCmd() *cobra.Command {
	return createRecoverCmd
}

func init() {
	flags.ApplyRecoverOpts(createRecoverCmd)
}

func buildRecoverJob() (*v1alpha1.Recover, error) {
	namespace := flags.GetDefaultVineyardNamespace()
	recover := &v1alpha1.Recover{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.RecoverName,
			Namespace: namespace,
		},
		Spec: v1alpha1.RecoverSpec{
			BackupName:      flags.BackupName,
			BackupNamespace: namespace,
		},
	}
	return recover, nil
}

// wait for the recover job to be done
func waitRecoverJobDone(c client.Client, recover *v1alpha1.Recover) error {
	return wait.PollImmediate(1*time.Second, 600*time.Second, func() (bool, error) {
		err := c.Get(context.TODO(), client.ObjectKey{
			Name:      recover.Name,
			Namespace: recover.Namespace,
		}, recover)
		if err != nil || recover.Status.State != k8s.SucceedState {
			return false, nil
		}
		return true, nil
	})
}
