/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

package legacy

import (
	"flag"
	"math/rand"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	v1alpha1 "github.com/v6d-io/v6d/k8s/api/k8s/v1alpha1"
	// +kubebuilder:scaffold:imports
)

var (
	scheme = runtime.NewScheme()
	// nolint: unused
	setupLog = ctrl.Log.WithName("scheduler")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

// nolint: unused
func startScheduler(channel chan struct{}) {
	command := app.NewSchedulerCommand(
		app.WithPlugin(Name, New),
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

// nolint: unused,deadcode
func main() {
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	rand.Seed(time.Now().UnixNano())

	scheduler := make(chan struct{})
	go startScheduler(scheduler)
	<-scheduler
}
