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

package k8s

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"
	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/operation"
)

// RecoverReconciler reconciles a Recover object
type RecoverReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Template kubernetes.Repo
}

// RecoverConfig holds all configuration about recover
type RecoverConfig struct {
	Name               string
	Namespace          string
	Replicas           int
	RecoverPath        string
	VineyarddNamespace string
	VineyarddName      string
	Endpoint           string
	VineyardSockPath   string
}

// Recover contains the configuration about recover
var Recover RecoverConfig

// GetRecoverConfig get recover configuratiin from Recover
func getRecoverConfig() RecoverConfig {
	return Recover
}

//+kubebuilder:rbac:groups=k8s.v6d.io,resources=recovers,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=k8s.v6d.io,resources=recovers/status,verbs=get;update;patch

func (r *RecoverReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithName("controllers").WithName("Recover")

	recover := k8sv1alpha1.Recover{}
	if err := r.Get(ctx, req.NamespacedName, &recover); err != nil {
		logger.Error(err, "unable to fetch Recover")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	logger.Info("Reconciling Recover", "recover", recover)

	app := kubernetes.Application{
		Client:   r.Client,
		FileRepo: r.Template,
		CR:       &recover,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Recover"),
		TmplFunc: map[string]interface{}{"getBackupConfig": getBackupConfig},
	}

	backup := k8sv1alpha1.Backup{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: recover.Spec.BackupNamespace, Name: recover.Spec.BackupName}, &backup); err != nil {
		logger.Error(err, "unable to get Backup")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// get vineyardd
	vineyardd := &k8sv1alpha1.Vineyardd{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: backup.Spec.VineyarddNamespace, Name: backup.Spec.VineyarddName}, vineyardd); err != nil {
		logger.Error(err, "unable to fetch Vineyardd")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	Recover.Name = "recover-" + backup.Name
	Recover.Namespace = backup.Namespace
	Recover.Replicas = vineyardd.Spec.Replicas
	Recover.RecoverPath = backup.Spec.BackupPath
	Recover.VineyarddName = backup.Spec.VineyarddName
	Recover.VineyarddNamespace = backup.Spec.VineyarddNamespace
	utils := operation.ClientUtils{Client: r.Client}
	socket, err := utils.ResolveRequiredVineyarddSocket(ctx, vineyardd.Name, vineyardd.Namespace, backup.Namespace)
	if err != nil {
		logger.Error(err, "unable to resolve vineyardd socket")
		return ctrl.Result{}, err
	}
	Recover.VineyardSockPath = socket
	Recover.Endpoint = backup.Spec.VineyarddName + "-rpc." + backup.Spec.VineyarddNamespace

	fmt.Println("Recover:", Recover)
	if _, err := app.Apply(ctx, "recover/job.yaml", logger, false); err != nil {
		logger.Error(err, "failed to apply recover job")
		return ctrl.Result{}, err
	}

	// reconcile every minute
	var duration, _ = time.ParseDuration("1m")
	return ctrl.Result{RequeueAfter: duration}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *RecoverReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.Recover{}).
		Complete(r)
}
