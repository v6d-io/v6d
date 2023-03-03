/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

// Package k8s contains k8s API versions.
package k8s

import (
	"context"
	"strconv"
	"time"

	"github.com/pkg/errors"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/operation"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// BackupConfig holds all configuration about backup
type BackupConfig struct {
	Limit              string
	Name               string
	Namespace          string
	Replicas           int
	BackupPath         string
	VineyarddNamespace string
	VineyarddName      string
	Endpoint           string
	VineyardSockPath   string
	Allinstances       string
}

// Backup contains the configuration about backup
var Backup BackupConfig

// GetBackupConfig get backup configuratiin from Backup
func getBackupConfig() BackupConfig {
	return Backup
}

func getResourceStorage(q resource.Quantity) string {
	return q.String()
}

// BackupReconciler reconciles a Backup object
type BackupReconciler struct {
	client.Client
	record.EventRecorder
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=backups,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=backups/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings,verbs=*

// Reconcile reconciles the Backup.
func (r *BackupReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithName("controllers").WithName("Backup")

	backup := k8sv1alpha1.Backup{}
	if err := r.Get(ctx, req.NamespacedName, &backup); err != nil {
		logger.Error(err, "unable to fetch Backup")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	logger.Info("Reconciling Backup", "backup", backup)

	// get vineyardd
	vineyardd := &k8sv1alpha1.Vineyardd{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: backup.Spec.VineyarddNamespace, Name: backup.Spec.VineyarddName}, vineyardd); err != nil {
		logger.Error(err, "unable to fetch Vineyardd")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	app := kubernetes.Application{
		Client:   r.Client,
		FileRepo: templates.Repo,
		CR:       &backup,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Backup"),
		TmplFunc: map[string]interface{}{
			"getResourceStorage": getResourceStorage,
			"getBackupConfig":    getBackupConfig,
		},
		Recorder: r.EventRecorder,
	}

	Backup.Name = "backup-" + backup.Spec.VineyarddName + "-" + backup.Spec.VineyarddNamespace
	Backup.Namespace = backup.Namespace
	Backup.Replicas = vineyardd.Spec.Replicas
	Backup.Limit = strconv.Itoa(backup.Spec.Limit)
	Backup.VineyarddName = backup.Spec.VineyarddName
	Backup.VineyarddNamespace = backup.Spec.VineyarddNamespace
	Backup.BackupPath = backup.Spec.BackupPath
	utils := operation.ClientUtils{Client: r.Client}
	socket, err := utils.ResolveRequiredVineyarddSocket(
		ctx,
		vineyardd.Name,
		vineyardd.Namespace,
		backup.Namespace,
	)
	if err != nil {
		logger.Error(err, "unable to resolve vineyardd socket")
		return ctrl.Result{}, err
	}
	Backup.VineyardSockPath = socket
	Backup.Endpoint = backup.Spec.VineyarddName + "-rpc." + backup.Spec.VineyarddNamespace
	Backup.Allinstances = strconv.Itoa(vineyardd.Spec.Replicas)

	if backup.Status.State == "" || backup.Status.State == RunningState {
		if _, err := app.Apply(ctx, "backup/job.yaml", logger, false); err != nil {
			logger.Error(err, "failed to apply backup job")
			return ctrl.Result{}, err
		}
		if _, err := app.Apply(ctx, "backup/backup-pv.yaml", logger, false); err != nil {
			logger.Error(err, "failed to apply backup pv")
			return ctrl.Result{}, err
		}
		if _, err := app.Apply(ctx, "backup/backup-pvc.yaml", logger, false); err != nil {
			logger.Error(err, "failed to apply backup pv")
			return ctrl.Result{}, err
		}
		if _, err := app.Apply(ctx, "backup/cluster-role.yaml", logger, true); err != nil {
			logger.Error(err, "failed to apply backup cluster role")
			return ctrl.Result{}, err
		}
		if _, err := app.Apply(ctx, "backup/cluster-role-binding.yaml", logger, true); err != nil {
			logger.Error(err, "failed to apply backup cluster role binding")
			return ctrl.Result{}, err
		}
		if err := r.UpdateStatus(ctx, &backup); err != nil {
			logger.Error(err, "failed to update status")
			return ctrl.Result{}, err
		}
	}

	// reconcile every minute
	duration, _ := time.ParseDuration("1m")
	return ctrl.Result{RequeueAfter: duration}, nil
}

// UpdateStatus updates the status of the Backup.Running
func (r *BackupReconciler) UpdateStatus(ctx context.Context, backup *k8sv1alpha1.Backup) error {
	name := client.ObjectKey{
		Name:      "backup-" + backup.Spec.VineyarddName + "-" + backup.Spec.VineyarddNamespace,
		Namespace: backup.Namespace,
	}
	job := batchv1.Job{}
	if err := r.Get(ctx, name, &job); err != nil {
		log.V(1).Error(err, "failed to get job")
		return err
	}

	state := RunningState
	if job.Status.Succeeded == *job.Spec.Parallelism {
		state = SucceedState
	}
	// get the running backup
	status := &k8sv1alpha1.BackupStatus{
		State: state,
	}
	if err := r.applyStatusUpdate(ctx, backup, status); err != nil {
		return errors.Wrap(err, "failed to update status")
	}
	return nil
}

func (r *BackupReconciler) applyStatusUpdate(ctx context.Context,
	backup *k8sv1alpha1.Backup, status *k8sv1alpha1.BackupStatus,
) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		name := client.ObjectKey{Name: backup.Name, Namespace: backup.Namespace}
		if err := r.Get(ctx, name, backup); err != nil {
			return errors.Wrap(err, "failed to get backup")
		}
		backup.Status = *status
		backup.Kind = "Backup"
		if err := kubernetes.ApplyOverlay(backup, &k8sv1alpha1.Backup{Status: *status}); err != nil {
			return errors.Wrap(err, "failed to overlay backup's status")
		}
		if err := r.Status().Update(ctx, backup); err != nil {
			return errors.Wrap(err, "failed to update backup's status")
		}
		return nil
	})
}

// SetupWithManager sets up the controller with the Manager.
func (r *BackupReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.Backup{}).
		Complete(r)
}
