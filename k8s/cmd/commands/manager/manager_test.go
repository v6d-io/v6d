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

// Package start contains the start command of vineyard operator
package manager

import (
	"testing"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	ctrl "sigs.k8s.io/controller-runtime"
)

/*func TestNewManagerCmd(t *testing.T) {
	tests := []struct {
		name string
		want *cobra.Command
	}{
		// TODO: Add test cases.
		{
			name: "Test Case 1",
			want: managerCmd, // 指定预期的 *cobra.Command 值
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewManagerCmd(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewManagerCmd() = %v, want %v", got, tt.want)
			}
		})
	}
}*/

// not implemented
/*func Test_startManager(t *testing.T) {
	mgr1, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 util.Scheme(),
		MetricsBindAddress:     flags.MetricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: flags.ProbeAddr,
		LeaderElection:         flags.EnableLeaderElection,
		LeaderElectionID:       "5fa514f1.v6d.io",
	})
	type args struct {
		mgr                  manager.Manager
		metricsAddr          string
		probeAddr            string
		enableLeaderElection bool
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
		{
			name: "case 1",
			args: args{
				mgr:                  mgr1,
				metricsAddr:          flags.MetricsAddr,
				probeAddr:            flags.ProbeAddr,
				enableLeaderElection: flags.EnableLeaderElection,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fmt.Println(flags.MetricsAddr)
			fmt.Println(flags.ProbeAddr)
			fmt.Println(flags.EnableLeaderElection)
			flags.Namespace = "vineyard-system"
			startManager(tt.args.mgr, tt.args.metricsAddr, tt.args.probeAddr, tt.args.enableLeaderElection)
		})
	}
}*/

func TestStartScheduler(t *testing.T) {
	// 初始化 Manager
	mgr, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 util.Scheme(),
		MetricsBindAddress:     flags.MetricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: flags.ProbeAddr,
		LeaderElection:         flags.EnableLeaderElection,
		LeaderElectionID:       "5fa514f1.v6d.io",
	})

	// 调度器配置文件路径
	schedulerConfigFile := "/home/zhuyi/v6d/k8s/config/scheduler/config.yaml"

	// 调用 startScheduler
	go startScheduler(mgr, schedulerConfigFile)

}

/*func Test_startScheduler(t *testing.T) {
	mgr, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 util.Scheme(),
		MetricsBindAddress:     flags.MetricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: flags.ProbeAddr,
		LeaderElection:         flags.EnableLeaderElection,
		LeaderElectionID:       "5fa514f1.v6d.io",
	})
	type args struct {
		mgr                 manager.Manager
		schedulerConfigFile string
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
		{
			name: "test case",
			args: args{
				mgr:                 mgr,
				schedulerConfigFile: "/home/zhuyi/v6d/k8s/config/scheduler/config.yaml",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startScheduler(tt.args.mgr, tt.args.schedulerConfigFile)
		})
	}
}*/
