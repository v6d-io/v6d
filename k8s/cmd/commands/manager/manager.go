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
	"log"
	"sync"

	"github.com/spf13/cobra"
	"go.uber.org/zap/zapcore"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	controllers "github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/schedulers"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
	"github.com/v6d-io/v6d/k8s/pkg/webhook/operation"
	"github.com/v6d-io/v6d/k8s/pkg/webhook/scheduling"
	"github.com/v6d-io/v6d/k8s/pkg/webhook/sidecar"
)

// managerCmd starts the manager of vineyard operator
var managerCmd = &cobra.Command{
	Use:   "manager",
	Short: "Start the manager of vineyard operator",
	Long: `Start the manager of vineyard operator.
For example:

# start the manager of vineyard operator with default configuration(Enable the controller, webhooks and scheduler)
vineyardctl manager

# start the manager of vineyard operator without webhooks
vineyardctl manager --enable-webhook false

# start the manager of vineyard operator without scheduler
vineyardctl manager --enable-scheduler false

# only start the controller
vineyardctl manager --enable-webhook false --enable-scheduler false`,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		opts := zap.Options{
			Development: true,
			TimeEncoder: zapcore.ISO8601TimeEncoder,
		}

		ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

		// start the controller
		mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
			Scheme:                 util.Scheme(),
			MetricsBindAddress:     flags.MetricsAddr,
			Port:                   9443,
			HealthProbeBindAddress: flags.ProbeAddr,
			LeaderElection:         flags.EnableLeaderElection,
			LeaderElectionID:       "5fa514f1.v6d.io",
		})
		if err != nil {
			util.ErrLogger.Fatal("unbale to setup the manager:", err)
		}

		wg := sync.WaitGroup{}

		if flags.EnableScheduler {
			wg.Add(1)
			go startScheduler(mgr, flags.SchedulerConfigFile)
		}

		wg.Add(1)
		go startManager(mgr, flags.MetricsAddr, flags.ProbeAddr, flags.EnableLeaderElection)

		wg.Wait()
	},
}

func NewManagerCmd() *cobra.Command {
	return managerCmd
}

func init() {
	flags.ApplyManagersOpts(managerCmd)
}

func startManager(
	mgr manager.Manager,
	metricsAddr string,
	probeAddr string,
	enableLeaderElection bool,
) {
	var err error

	if err = (&controllers.LocalObjectReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal(
			"unable to create controller",
			"controller",
			"LocalObject",
			"error: ",
			err,
		)
	}
	if err = (&controllers.GlobalObjectReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal(
			"unable to create controller",
			"controller",
			"GlobalObject",
			"error: ",
			err,
		)
	}
	if err = (&controllers.VineyarddReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("vineyardd-controller"),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal(
			"unable to create controller",
			"controller",
			"Vineyardd",
			"error: ",
			err,
		)
	}
	if err = (&controllers.OperationReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("operation-controller"),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal(
			"unable to create controller",
			"controller",
			"Operation",
			"error: ",
			err,
		)
	}

	if err = (&controllers.SidecarReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("sidecar-controller"),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal("unable to create controller", "controller", "Sidecar", "error: ", err)
	}
	if err = (&controllers.BackupReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("backup-controller"),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal("unable to create controller", "controller", "Backup", "error: ", err)
	}

	if err = (&controllers.RecoverReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
	}).SetupWithManager(mgr); err != nil {
		util.ErrLogger.Fatal("unable to create controller", "controller", "Recover", "error: ", err)
	}

	if !flags.EnableWebhook {
		// register the webhooks of CRDs
		if err = (&v1alpha1.LocalObject{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal(
				"unable to create webhook",
				"webhook",
				"LocalObject",
				"error: ",
				err,
			)
		}
		if err = (&v1alpha1.GlobalObject{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal(
				"unable to create webhook",
				"webhook",
				"GlobalObject",
				"error: ",
				err,
			)
		}
		if err = (&v1alpha1.Vineyardd{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal("unable to create webhook", "webhook", "Vineyardd", "error: ", err)
		}
		if err = (&v1alpha1.Operation{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal("unable to create webhook", "webhook", "Operation", "error: ", err)
		}
		if err = (&v1alpha1.Sidecar{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal("unable to create webhook", "webhook", "Sidecar", "error: ", err)
		}
		if err = (&v1alpha1.Backup{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal("unable to create webhook", "webhook", "Backup", "error: ", err)
		}
		if err = (&v1alpha1.Recover{}).SetupWebhookWithManager(mgr); err != nil {
			util.ErrLogger.Fatal("unable to create webhook", "webhook", "Recover", "error: ", err)
		}

		// register the assembly webhook
		log.Println("registering the assembly webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod",
			&webhook.Admission{
				Handler: &operation.AssemblyInjector{Client: mgr.GetClient()},
			})
		log.Println("the assembly webhook is registered")

		// register the sidecar webhook
		log.Println("registering the sidecar webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod-sidecar",
			&webhook.Admission{
				Handler: &sidecar.Injector{
					Client:   mgr.GetClient(),
					Template: templates.NewEmbedTemplate(),
				},
			})
		log.Println("the sidecar webhook is registered")

		// register the scheduling webhook
		log.Println("registering the scheduling webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod-scheduling",
			&webhook.Admission{
				Handler: &scheduling.Injector{Client: mgr.GetClient()},
			})
		log.Println("the scheduling webhook is registered")

		if err := mgr.AddHealthzCheck("healthz", mgr.GetWebhookServer().StartedChecker()); err != nil {
			util.ErrLogger.Fatal("unable to set up health check for webhook: ", err)
		}
		if err := mgr.AddReadyzCheck("readyz", mgr.GetWebhookServer().StartedChecker()); err != nil {
			util.ErrLogger.Fatal("unable to set up ready check for webhook: ", err)
		}
	} else {
		if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
			util.ErrLogger.Fatal("unable to set up health check: ", err)
		}
		if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
			util.ErrLogger.Fatal("unable to set up ready check: ", err)
		}
	}

	log.Println("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		util.ErrLogger.Fatal("problem running manager: ", err)
	}
}

func startScheduler(mgr manager.Manager, schedulerConfigFile string) {
	command := app.NewSchedulerCommand(
		app.WithPlugin(schedulers.Name,
			func(obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
				return schedulers.New(mgr.GetClient(), mgr.GetConfig(), obj, handle)
			},
		),
	)

	args := make([]string, 0, 10)
	// apply logs
	args = append(args, "-v")
	args = append(args, "5")
	// apply scheduler plugin config
	args = append(args, "--config")
	args = append(args, flags.SchedulerConfigFile)
	command.SetArgs(args)
	if err := command.Execute(); err != nil {
		util.ErrLogger.Fatal("problem running scheduler: ", err)
	}
}
