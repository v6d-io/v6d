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
package util

import (
	"os"
	"testing"
)

func Test_ReadFromFile(t *testing.T) {
	tempFileContent := ""
	tempFile, err := os.CreateTemp("", "testfile")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}
	defer tempFile.Close()
	if err = os.WriteFile(tempFile.Name(), []byte(tempFileContent), 0644); err != nil {
		t.Errorf("Failed to write file: %v", err)
	}

	type args struct {
		path string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				path: tempFile.Name(),
			},
			want:    tempFileContent,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReadFromFile(tt.args.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReadFromFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ReadFromFile() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_ReadFromStdin(t *testing.T) {
	type args struct {
		args []string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				args: []string{"-"},
			},
			want:    "Test input",
			wantErr: false,
		},
	}

	expected := "Test input"
	r, w, _ := os.Pipe()
	os.Stdin = r
	go func() {
		defer w.Close()
		if _, err := w.Write([]byte(expected)); err != nil {
			t.Errorf("Failed to write file: %v", err)
		}
	}()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReadFromStdin(tt.args.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReadFromStdin() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ReadFromStdin() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_ReadJsonFromStdin(t *testing.T) {
	type args struct {
		args []string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "Test case 1",
			args: args{
				args: []string{"-"},
			},
			want:    `{"test": "input"}`,
			wantErr: false,
		},
	}

	expected := `{"test": "input"}`
	r, w, _ := os.Pipe()
	os.Stdin = r
	go func() {
		defer w.Close()
		if _, err := w.Write([]byte(expected)); err != nil {
			t.Errorf("Failed to write file: %v", err)
		}
	}()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReadJsonFromStdin(tt.args.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReadJsonFromStdin() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ReadJsonFromStdin() = %v, want %v", got, tt.want)
			}
		})
	}
}
