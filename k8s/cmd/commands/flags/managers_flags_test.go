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
)

/*func TestApplyManagersOpts(t *testing.T) {
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
			ApplyManagersOpts(tt.args.cmd)
		})
	}
}*/

func TestApplyManagersOpts(t *testing.T) {
	cmd := &cobra.Command{}
	ApplyManagersOpts(cmd)

	t.Run("DefaultValues", func(t *testing.T) {
		// 执行命令
		err := cmd.Execute()

		// 检查默认值
		if MetricsAddr != "127.0.0.1:8080" {
			t.Errorf("Default metrics bind address is incorrect, got: %s, want: 127.0.0.1:8080", MetricsAddr)
		}
		if ProbeAddr != ":8081" {
			t.Errorf("Default health probe bind address is incorrect, got: %s, want: :8081", ProbeAddr)
		}
		if EnableLeaderElection != false {
			t.Errorf("Default leader election value is incorrect, got: %t, want: false", EnableLeaderElection)
		}
		if EnableWebhook != true {
			t.Errorf("Default enable webhook value is incorrect, got: %t, want: true", EnableWebhook)
		}
		if EnableScheduler != true {
			t.Errorf("Default enable scheduler value is incorrect, got: %t, want: true", EnableScheduler)
		}
		if SchedulerConfigFile != "/etc/kubernetes/scheduler.yaml" {
			t.Errorf("Default scheduler config file is incorrect, got: %s, want: /etc/kubernetes/scheduler.yaml", SchedulerConfigFile)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})

	t.Run("CustomValues", func(t *testing.T) {
		// 设置自定义参数值
		cmd.SetArgs([]string{
			"--metrics-bind-address", "10.0.0.1:9090",
			"--health-probe-bind-address", ":8888",
			"--leader-elect",
			"--enable-webhook=false",
			"--enable-scheduler=false",
			"--scheduler-config-file", "/path/to/custom/scheduler.yaml",
		})

		// 执行命令
		err := cmd.Execute()

		// 检查自定义值
		if MetricsAddr != "10.0.0.1:9090" {
			t.Errorf("Custom metrics bind address is incorrect, got: %s, want: 10.0.0.1:9090", MetricsAddr)
		}
		if ProbeAddr != ":8888" {
			t.Errorf("Custom health probe bind address is incorrect, got: %s, want: :8888", ProbeAddr)
		}
		if EnableLeaderElection != true {
			t.Errorf("Custom leader election value is incorrect, got: %t, want: true", EnableLeaderElection)
		}
		if EnableWebhook != false {
			t.Errorf("Custom enable webhook value is incorrect, got: %t, want: false", EnableWebhook)
		}
		if EnableScheduler != false {
			t.Errorf("Custom enable scheduler value is incorrect, got: %t, want: false", EnableScheduler)
		}
		if SchedulerConfigFile != "/path/to/custom/scheduler.yaml" {
			t.Errorf("Custom scheduler config file is incorrect, got: %s, want: /path/to/custom/scheduler.yaml", SchedulerConfigFile)
		}

		// 检查是否有错误发生
		if err != nil {
			t.Errorf("Error executing command: %v", err)
		}
	})
}
