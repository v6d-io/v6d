/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

package main

import (
	"flag"
	"math/rand"
	"os"
	"time"

	"go.uber.org/zap/zapcore"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	controllers "github.com/v6d-io/v6d/k8s/controllers/k8s"
	k8scontrollers "github.com/v6d-io/v6d/k8s/controllers/k8s"
	"github.com/v6d-io/v6d/k8s/operator"
	"github.com/v6d-io/v6d/k8s/schedulers"
	"github.com/v6d-io/v6d/k8s/templates"
	// +kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("manager")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(k8sv1alpha1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

func startManager(channel chan struct{}, mgr manager.Manager, metricsAddr string, probeAddr string, enableLeaderElection bool) {
	var err error

	if err = (&controllers.LocalObjectReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("localobject-controller"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "LocalObject")
		os.Exit(1)
	}
	if err = (&controllers.GlobalObjectReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("globalobject-controller"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GlobalObject")
		os.Exit(1)
	}
	if err = (&k8scontrollers.VineyarddReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Template: templates.NewEmbedTemplate(),
		Recorder: mgr.GetEventRecorderFor("vineyardd-controller"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Vineyardd")
		os.Exit(1)
	}
	if os.Getenv("ENABLE_WEBHOOKS") != "false" {
		if err = (&k8sv1alpha1.LocalObject{}).SetupWebhookWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to create webhook", "webhook", "LocalObject")
			os.Exit(1)
		}
		if err = (&k8sv1alpha1.GlobalObject{}).SetupWebhookWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to create webhook", "webhook", "GlobalObject")
			os.Exit(1)
		}
		if err = (&k8sv1alpha1.Vineyardd{}).SetupWebhookWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to create webhook", "webhook", "Vineyardd")
			os.Exit(1)
		}
		// register the assembly webhook
		setupLog.Info("registering the assembly webhook")
		mgr.GetWebhookServer().Register("/mutate-v1-pod",
			&webhook.Admission{
				Handler: &operator.AssemblyInjector{Client: mgr.GetClient()}})
		setupLog.Info("the assembly webhook is registered")

		if err := mgr.AddHealthzCheck("healthz", mgr.GetWebhookServer().StartedChecker()); err != nil {
			setupLog.Error(err, "unable to set up health check for webhook")
			os.Exit(1)
		}
		if err := mgr.AddReadyzCheck("readyz", mgr.GetWebhookServer().StartedChecker()); err != nil {
			setupLog.Error(err, "unable to set up ready check for webhook")
			os.Exit(1)
		}
	}
	// +kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func startScheduler(channel chan struct{}, mgr manager.Manager) {
	command := app.NewSchedulerCommand(
		app.WithPlugin(schedulers.Name,
			func(obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
				return schedulers.New(mgr.GetClient(), mgr.GetConfig(), obj, handle)
			},
		),
	)

	args := make([]string, 4)
	args[0] = "-v"
	args[1] = "5"
	args[2] = "--config"
	args[3] = "/etc/kubernetes/scheduler.yaml"
	command.SetArgs(args)
	if err := command.Execute(); err != nil {
		setupLog.Error(err, "problem running scheduler")
		os.Exit(1)
	}
	close(channel)
}

func main() {
	var metricsAddr string
	var enableLeaderElection bool
	var probeAddr string
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	opts := zap.Options{
		Development: true,
		TimeEncoder: zapcore.ISO8601TimeEncoder,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	rand.Seed(time.Now().UnixNano())
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     metricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "5fa514f1.v6d.io",
	})
	if err != nil {
		setupLog.Error(err, "unable to setup the manager")
		os.Exit(1)
	}

	scheduler := make(chan struct{})
	go startScheduler(scheduler, mgr)

	manager := make(chan struct{})
	go startManager(manager, mgr, metricsAddr, probeAddr, enableLeaderElection)

	<-scheduler
	<-manager
}
