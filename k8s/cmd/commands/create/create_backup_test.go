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
	"fmt"
	"os"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
)

func Test_BuildBackup(t *testing.T) {
	flags.KubeConfig = os.Getenv("KUBECONFIG")
	flags.BackupName = "test-backup"
	flags.Namespace = "test_backup"
	flags.BackupOpts.BackupPath = "backup/path/to/test"
	c := util.KubernetesClient()

	type args struct {
		c    client.Client
		args []string
	}
	tests := []struct {
		name    string
		args    args
		want    *v1alpha1.Backup
		wantErr bool
	}{
		{
			name: "Valid JSON from stdin",
			args: args{
				c:    c,
				args: []string{},
			},
			want: &v1alpha1.Backup{
				// Expected Backup CR based on the provided JSON.
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-backup",
					Namespace: "test_backup",
				},
				Spec: v1alpha1.BackupSpec{
					BackupPath: "backup/path/to/test",
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildBackup(tt.args.c, tt.args.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildBackup() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			a := fmt.Sprint(got)
			b := fmt.Sprint(tt.want)
			if !reflect.DeepEqual(a, b) {
				t.Errorf("BuildBackup() = %v, want %v", got, tt.want)
			}
		})
	}
}
