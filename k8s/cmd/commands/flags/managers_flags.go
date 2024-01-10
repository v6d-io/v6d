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
	"github.com/spf13/cobra"
)

var (
	// MetricsAddr is the TCP address that the controller should bind to
	// for serving prometheus metrics.
	MetricsAddr string

	// ProbeAddr is the TCP address that the controller should bind to
	// for serving health probes.
	ProbeAddr string

	// EnableLeaderElection for controller manager.
	// Enabling this will ensure there is only one active controller manager.
	EnableLeaderElection bool

	// EnableWebhook will enable webhook for controller manager.
	EnableWebhook bool

	// EnableScheduler will enable scheduler for controller manager.
	EnableScheduler bool

	// SchedulerConfigFile is the location of scheduler plugin's configuration file.
	SchedulerConfigFile string

	// WebhookCertDir is the directory to store the generated certificates.
	WebhookCertDir string
)

func ApplyManagersOpts(cmd *cobra.Command) {
	cmd.Flags().
		StringVarP(&MetricsAddr, "metrics-bind-address", "", "127.0.0.1:8080",
			"The address the metric endpoint binds to.")
	cmd.Flags().
		StringVarP(&ProbeAddr, "health-probe-bind-address", "", ":8081",
			"The address the probe endpoint binds to.")
	cmd.Flags().BoolVarP(&EnableLeaderElection, "leader-elect", "", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	cmd.Flags().
		BoolVarP(&EnableWebhook, "enable-webhook", "", true,
			"Enable webhook for controller manager.")
	cmd.Flags().
		BoolVarP(&EnableScheduler, "enable-scheduler", "", true,
			"Enable scheduler for controller manager.")
	cmd.Flags().
		StringVarP(&SchedulerConfigFile, "scheduler-config-file", "",
			"/etc/kubernetes/scheduler.yaml",
			"The location of scheduler plugin's configuration file.")
	cmd.Flags().
		StringVarP(&WebhookCertDir, "webhook-cert-dir", "", "/etc/webhook/certs",
			"The directory to store the generated certificates.")
}
