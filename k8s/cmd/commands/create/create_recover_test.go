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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
)

func Test_BuildV1alphaRecoverCR(t *testing.T) {
	flags.RecoverName = "test-recover"
	flags.BackupName = "test-backup"

	tests := []struct {
		name    string
		want    *v1alpha1.Recover
		wantErr bool
	}{
		// Add test cases.
		{
			name: "Test Case 1",
			want: &v1alpha1.Recover{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-recover",
					Namespace: flags.GetDefaultVineyardNamespace(),
				},
				Spec: v1alpha1.RecoverSpec{
					BackupName:      "test-backup",
					BackupNamespace: flags.GetDefaultVineyardNamespace(),
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildV1alphaRecoverCR()
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildV1alphaRecoverCR() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BuildV1alphaRecoverCR() = %v, want %v", got, tt.want)
			}
		})
	}
}
