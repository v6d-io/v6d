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
package flags

import (
	"testing"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"os"
)

/*func Test_defaultKubeConfig(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := defaultKubeConfig(); got != tt.want {
				t.Errorf("defaultKubeConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}*/

func TestDefaultKubeConfig(t *testing.T) {
	// 设置环境变量 KUBECONFIG 和 HOME
	os.Setenv("KUBECONFIG", "")
	os.Setenv("HOME", "/home/user")

	t.Run("EmptyKubeConfigEnvVar", func(t *testing.T) {
		kubeconfig := defaultKubeConfig()

		// 检查默认情况下 kubeconfig 的值
		expected := "/home/user/.kube/config"
		if kubeconfig != expected {
			t.Errorf("Default kubeconfig is incorrect, got: %s, want: %s", kubeconfig, expected)
		}
	})

	t.Run("NonEmptyKubeConfigEnvVar", func(t *testing.T) {
		// 设置 KUBECONFIG 环境变量
		os.Setenv("KUBECONFIG", "/path/to/custom/config")

		kubeconfig := defaultKubeConfig()

		// 检查指定 KUBECONFIG 环境变量后 kubeconfig 的值
		expected := "/path/to/custom/config"
		if kubeconfig != expected {
			t.Errorf("Custom kubeconfig is incorrect, got: %s, want: %s", kubeconfig, expected)
		}

		// 清除 KUBECONFIG 环境变量
		os.Setenv("KUBECONFIG", "")
	})

	t.Run("NonEmptyKubeConfigEnvVarWithHome", func(t *testing.T) {
		// 设置 KUBECONFIG 和 HOME 环境变量
		os.Setenv("KUBECONFIG", "/path/to/custom/config")
		os.Setenv("HOME", "/home/otheruser")

		kubeconfig := defaultKubeConfig()

		// 检查指定 KUBECONFIG 和 HOME 环境变量后 kubeconfig 的值
		expected := "/path/to/custom/config"
		if kubeconfig != expected {
			t.Errorf("Custom kubeconfig with HOME is incorrect, got: %s, want: %s", kubeconfig, expected)
		}

		// 清除 KUBECONFIG 和 HOME 环境变量
		os.Setenv("KUBECONFIG", "")
		os.Setenv("HOME", "/home/user")
	})
}

/*func TestGetDefaultVineyardNamespace(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetDefaultVineyardNamespace(); got != tt.want {
				t.Errorf("GetDefaultVineyardNamespace() = %v, want %v", got, tt.want)
			}
		})
	}
}*/

func TestGetDefaultVineyardNamespace(t *testing.T) {
	// 设置 Namespace 全局变量的值
	Namespace = "default-namespace"

	t.Run("DefaultNamespace", func(t *testing.T) {
		namespace := GetDefaultVineyardNamespace()

		// 检查默认情况下 Namespace 的值
		expected := "default-namespace"
		if namespace != expected {
			t.Errorf("Default namespace is incorrect, got: %s, want: %s", namespace, expected)
		}
	})

	// 恢复 Namespace 全局变量的原始值
	Namespace = ""
}

/*func TestApplyGlobalFlags(t *testing.T) {
	type args struct {
		cmd *cobra.Command
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ApplyGlobalFlags(tt.args.cmd)
		})
	}
}*/

func TestApplyGlobalFlags(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyGlobalFlags(cmd)

	t.Run("DefaultValues", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认值
		if KubeConfig != defaultKubeConfig() {
			t.Errorf("Default kubeconfig is incorrect, got: %s, want: %s", KubeConfig, defaultKubeConfig())
		}
		if Namespace != defaultNamespace {
			t.Errorf("Default namespace is incorrect, got: %s, want: %s", Namespace, defaultNamespace)
		}
		if Wait != true {
			t.Errorf("Default wait value is incorrect, got: %t, want: true", Wait)
		}
		if CreateNamespace != false {
			t.Errorf("Default create namespace value is incorrect, got: %t, want: false", CreateNamespace)
		}
		if DumpUsage != false {
			t.Errorf("Default dump usage value is incorrect, got: %t, want: false", DumpUsage)
		}
		if GenDoc != false {
			t.Errorf("Default gen doc value is incorrect, got: %t, want: false", GenDoc)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomValues", func(t *testing.T) {
		// 设置自定义参数值
		cmd.SetArgs([]string{
			"--kubeconfig", "/path/to/custom/config",
			"--namespace", "custom-namespace",
			"--wait=false",
			"--create-namespace=true",
			"--dump-usage",
			"--gen-doc",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义值
		if KubeConfig != "/path/to/custom/config" {
			t.Errorf("Custom kubeconfig is incorrect, got: %s, want: /path/to/custom/config", KubeConfig)
		}
		if Namespace != "custom-namespace" {
			t.Errorf("Custom namespace is incorrect, got: %s, want: custom-namespace", Namespace)
		}
		if Wait != false {
			t.Errorf("Custom wait value is incorrect, got: %t, want: false", Wait)
		}
		if CreateNamespace != true {
			t.Errorf("Custom create namespace value is incorrect, got: %t, want: true", CreateNamespace)
		}
		if DumpUsage != true {
			t.Errorf("Custom dump usage value is incorrect, got: %t, want: true", DumpUsage)
		}
		if GenDoc != true {
			t.Errorf("Custom gen doc value is incorrect, got: %t, want: true", GenDoc)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}

/*func TestRemoveVersionFlag(t *testing.T) {
	type args struct {
		f *pflag.FlagSet
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RemoveVersionFlag(tt.args.f)
		})
	}
}*/

func TestRemoveVersionFlag(t *testing.T) {
	// 创建一个 FlagSet
	fs := pflag.NewFlagSet("test", pflag.ExitOnError)

	// 添加一个自定义的 "version" 标志
	versionFlag := fs.String("version", "", "Custom version flag")

	t.Run("FlagRemoval", func(t *testing.T) {
		// 调用 RemoveVersionFlag 函数
		RemoveVersionFlag(fs)

		// 检查标志是否被正确修改
		if fs.Lookup("version") != nil {
			t.Error("Version flag still exists in FlagSet after removal")
		}
		if fs.Lookup("x-version") == nil {
			t.Error("x-version flag not found in FlagSet after removal")
		}
	})

	t.Run("CommandLineParsing", func(t *testing.T) {
		// 重置标志的值
		*versionFlag = ""

		// 创建一个新的 FlagSet 用于命令行参数解析
		newFS := pflag.NewFlagSet("testCmd", pflag.ExitOnError)
		// 将原始 FlagSet 中的标志复制到新的 FlagSet
		fs.VisitAll(func(f *pflag.Flag) {
			newFS.AddFlag(f)
		})

		// 设置命令行参数
		args := []string{"--x-version", "1.0"}

		// 解析命令行参数
		err := newFS.Parse(args)

		// 检查解析结果和标志的值
		if err != nil {
			t.Errorf("Error parsing command line arguments: %v", err)
		}
		if *versionFlag != "1.0" {
			t.Errorf("Version flag value is incorrect, got: %s, want: 1.0", *versionFlag)
		}
	})
}

/*func TestRestoreVersionFlag(t *testing.T) {
	type args struct {
		f *pflag.FlagSet
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RestoreVersionFlag(tt.args.f)
		})
	}
}*/

func TestRestoreVersionFlag(t *testing.T) {
	fs := pflag.NewFlagSet("test", pflag.ExitOnError)
	fs.String("x-version", "", "Custom version flag")

	t.Run("FlagRestoration", func(t *testing.T) {
		normalize := fs.GetNormalizeFunc()
		RestoreVersionFlag(fs)
		if fs.Lookup("x-version") != nil {
			t.Error("x-version flag still exists in FlagSet after restoration")
		}
		if fs.Lookup("version") == nil {
			t.Error("version flag not found in FlagSet after restoration")
		}
		fs.SetNormalizeFunc(normalize)
	})

	t.Run("CommandLineParsing", func(t *testing.T) {
		newFS := pflag.NewFlagSet("testCmd", pflag.ExitOnError)
		versionFlag := ""
		newFS.StringVar(&versionFlag, "version", "", "Custom version flag")
		args := []string{"--version", "1.0"}
		err := newFS.Parse(args)
		if err != nil {
			t.Errorf("Error parsing command line arguments: %v", err)
		}
		if versionFlag != "1.0" {
			t.Errorf("Version flag value is incorrect, got: %s, want: 1.0", versionFlag)
		}
	})
}
