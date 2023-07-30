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
	"reflect"
	"testing"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

/*func TestNewCreateBackupCmd(t *testing.T) {
	tests := []struct {
		name string
		want *cobra.Command
	}{
		{
			name: "EmptyArgs",
			want: createBackupCmd,
		},
		{
			name: "WithArgs",
			want: createBackupCmd,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewCreateBackupCmd(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewCreateBackupCmd() = %v, want %v", got, tt.want)
			}
		})
	}
}*/

func Test_buildBackupCR(t *testing.T) {
	want := &v1alpha1.Backup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      flags.BackupName,
			Namespace: flags.Namespace,
		},
		Spec: flags.BackupOpts,
	}

	got, err := buildBackupCR()

	if err != nil {
		t.Errorf("buildBackupCR() error = %v, wantErr false", err)
		return
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("buildBackupCR() = %v, want %v", got, want)
	}
}

func TestBuildBackup(t *testing.T) {
	type args struct {
		c    client.Client
		args []string
	}

	flags.KubeConfig = "/home/zhuyi/.kube/config"
	c := util.KubernetesClient()

	tests := []struct {
		name    string
		args    args
		want    *v1alpha1.Backup
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name: "Valid JSON from stdin",
			args: args{
				c:    c,
				args: []string{},
			},
			want: &v1alpha1.Backup{
				// Expected Backup CR based on the provided JSON.
				ObjectMeta: metav1.ObjectMeta{
					Name:      flags.BackupName,
					Namespace: flags.Namespace,
				},
				Spec: flags.BackupOpts,
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
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BuildBackup() = %v, want %v", got, tt.want)
			}
		})
	}
}
