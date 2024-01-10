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

// Package manager contains the start command of vineyard operator
package manager

import (
	"fmt"
	"sync"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/cmd/commands/util"
	controllers "github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/schedulers"
	"github.com/v6d-io/v6d/k8s/pkg/webhook/operation"
	"github.com/v6d-io/v6d/k8s/pkg/webhook/scheduling"
	"github.com/v6d-io/v6d/k8s/pkg/webhook/sidecar"
)

var managerExample = util.Examples(`
	# start the manager of vineyard operator with default configuration
	# (Enable the controller, webhooks and scheduler)
	vineyardctl manager

	# start the manager of vineyard operator without webhooks
	vineyardctl manager --enable-webhook=false

	# start the manager of vineyard operator without scheduler
	vineyardctl manager --enable-scheduler=false

	# only start the controller
	vineyardctl manager --enable-webhook=false --enable-scheduler=false`)

// managerCmd starts the manager of vineyard operator
var managerCmd = &cobra.Command{
	Use:     "manager",
	Short:   "Start the manager of vineyard operator",
	Example: managerExample,
	Run: func(cmd *cobra.Command, args []string) {
		util.AssertNoArgs(cmd, args)

		ctrl.SetLogger(log.Log.Logger)
		// start the controller
		mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
			Scheme:                 util.Scheme(),
			MetricsBindAddress:     flags.MetricsAddr,
			CertDir:                flags.WebhookCertDir,
			Port:                   9443,
			HealthProbeBindAddress: flags.ProbeAddr,
			LeaderElection:         flags.EnableLeaderElection,
			LeaderElectionID:       "5fa514f1.v6d.io",
		})
		if err != nil {
			log.Fatal(err, "unable to setup the manager")
		}

		wg := sync.WaitGroup{}

		if flags.EnableScheduler {
			wg.Add(1)
			go startScheduler(mgr, flags.SchedulerConfigFile)
		}

		wg.Add(1)
		go startManager(mgr, flags.MetricsAddr, flags.ProbeAddr,
			flags.EnableLeaderElection)

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
	clientset, err := kubernetes.NewForConfig(mgr.GetConfig())
	if err != nil {
		log.Fatal(err, "unable to create REST client")
	}

	if err := (&controllers.LocalObjectReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create local object controller")
	}
	if err := (&controllers.GlobalObjectReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create global object controller")
	}
	if err := (&controllers.VineyarddReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		EventRecorder: mgr.GetEventRecorderFor("vineyardd-controller"),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create vineyardd controller")
	}
	if err := (&controllers.OperationReconciler{
		Client:        mgr.GetClient(),
		Clientset:     clientset,
		Scheme:        mgr.GetScheme(),
		EventRecorder: mgr.GetEventRecorderFor("operation-controller"),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create operation controller")
	}

	if err := (&controllers.SidecarReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		EventRecorder: mgr.GetEventRecorderFor("sidecar-controller"),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create sidecar controller")
	}
	if err := (&controllers.BackupReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		EventRecorder: mgr.GetEventRecorderFor("backup-controller"),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create backup controller")
	}

	if err := (&controllers.RecoverReconciler{
		Client:    mgr.GetClient(),
		Clientset: clientset,
		Scheme:    mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create recover controller")
	}

	if err := (&controllers.CSIDriverReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		EventRecorder: mgr.GetEventRecorderFor("csidriver-controller"),
	}).SetupWithManager(mgr); err != nil {
		log.Fatal(err, "unable to create csidriver controller")
	}

	if flags.EnableWebhook {
		// register the webhooks of CRDs
		if err := (&v1alpha1.LocalObject{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create local object webhook")
		}
		if err := (&v1alpha1.GlobalObject{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create global object webhook")
		}
		if err := (&v1alpha1.Vineyardd{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create vineyardd webhook")
		}
		if err := (&v1alpha1.Operation{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create operator webhook")
		}
		if err := (&v1alpha1.Sidecar{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create sidecar webhook")
		}
		if err := (&v1alpha1.Backup{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create backup webhook")
		}
		if err := (&v1alpha1.Recover{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create recover webhook")
		}
		if err := (&v1alpha1.CSIDriver{}).SetupWebhookWithManager(mgr); err != nil {
			log.Fatal(err, "unable to create csidriver webhook")
		}

		// register the assembly webhook
		log.Info("registering the assembly webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod",
			&webhook.Admission{
				Handler: &operation.AssemblyInjector{Client: mgr.GetClient()},
			})
		log.Info("the assembly webhook is registered")

		// register the sidecar webhook
		log.Info("registering the sidecar webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod-sidecar",
			&webhook.Admission{
				Handler: &sidecar.Injector{
					Client: mgr.GetClient(),
				},
			})
		log.Info("the sidecar webhook is registered")

		// register the scheduling webhook
		log.Info("registering the scheduling webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod-scheduling",
			&webhook.Admission{
				Handler: &scheduling.Injector{Client: mgr.GetClient()},
			})
		log.Info("the scheduling webhook is registered")

		if err := mgr.AddHealthzCheck("healthz", mgr.GetWebhookServer().StartedChecker()); err != nil {
			log.Fatal(err, "unable to set up health check for webhook")
		}
		if err := mgr.AddReadyzCheck("readyz", mgr.GetWebhookServer().StartedChecker()); err != nil {
			log.Fatal(err, "unable to set up ready check for webhook")
		}
	} else {
		if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
			log.Fatal(err, "unable to set up health check")
		}
		if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
			log.Fatal(err, "unable to set up ready check")
		}
	}

	webhookDNSName := fmt.Sprintf("vineyard-webhook-service.%s.svc", flags.Namespace)
	certGenerator, err := util.NewCertGenerator(util.CACommonName, util.CAOrgainzation,
		webhookDNSName, flags.WebhookCertDir)
	if err != nil {
		log.Fatal(err, "unable to build certificates generator")
	}
	if err := certGenerator.Generate(); err != nil {
		log.Fatal(err, "unable to generate certificates with generator")
	}
	if err := certGenerator.PatchCABundleToMutatingWebhook("vineyard-mutating-webhook-configuration"); err != nil {
		log.Fatal(err, "unable to patch CAbundle to mutating webhook configuration")
	}
	if err := certGenerator.PatchCABundleToValidatingWebhook("vineyard-validating-webhook-configuration"); err != nil {
		log.Fatal(err, "unable to patch CAbundle to validating webhook configuration")
	}

	log.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		log.Fatal(err, "problem running manager")
	}
}

func startScheduler(mgr manager.Manager, schedulerConfigFile string) {
	// restore the global "x-version" flag back to "version" in `verflag`
	flags.RestoreVersionFlag(pflag.CommandLine)

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
		log.Fatal(err, "failed to execute scheduler")
	}
}
