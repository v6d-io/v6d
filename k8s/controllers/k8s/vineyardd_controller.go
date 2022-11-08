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

// Package k8s contains all controllers in the vineyard operator.
package k8s

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"
	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// VineyarddReconciler reconciles a Vineyardd object
type VineyarddReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Template kubernetes.Repo
	Recorder record.EventRecorder
}

// EtcdConfig holds all configuration about etcd
type EtcdConfig struct {
	Namespace string
	Rank      int
	Endpoints string
}

// Etcd contains the configuration about etcd
var Etcd EtcdConfig

// Get etcd configuratiin from Etcd
func getEtcdConfig() EtcdConfig {
	return Etcd
}

func getStorage(q resource.Quantity) string {
	return q.String()
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=services;serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings,verbs=*
// +kubebuilder:rbac:groups="",resources=persistentvolumes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete

// Reconcile reconciles the Vineyardd.
func (r *VineyarddReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)
	vineyardd := k8sv1alpha1.Vineyardd{}
	if err := r.Get(ctx, req.NamespacedName, &vineyardd); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	ctrl.Log.V(1).Info("Reconciling Vineyardd", "vineyardd", vineyardd)
	vineyarddFile, err := r.Template.GetFilesRecursive("vineyardd")
	if err != nil {
		ctrl.Log.Error(err, "failed to load vineyardd templates")
		return ctrl.Result{}, err
	}

	// deploy the vineyardd
	vineyarddApp := kubernetes.Application{
		Client:   r.Client,
		FileRepo: r.Template,
		CR:       &vineyardd,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Vineyardd"),
		Recorder: r.Recorder,
		TmplFunc: map[string]interface{}{"getStorage": getStorage},
	}
	etcdApp := kubernetes.Application{
		Client:   r.Client,
		FileRepo: r.Template,
		CR:       &vineyardd,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Vineyardd"),
		TmplFunc: map[string]interface{}{"getEtcdConfig": getEtcdConfig},
	}
	// set up the etcd
	Etcd.Namespace = vineyardd.Namespace
	Etcd.Endpoints = ""
	replicas := vineyardd.Spec.Etcd.Replicas
	for i := 0; i < replicas; i++ {
		Etcd.Endpoints = Etcd.Endpoints + "etcd" + strconv.Itoa(i) + "=http://etcd" + strconv.Itoa(i) + ":2380,"
	}
	Etcd.Endpoints = Etcd.Endpoints[:len(Etcd.Endpoints)-1]

	for i := 0; i < replicas; i++ {
		Etcd.Rank = i
		if _, err := etcdApp.Apply(ctx, "etcd/etcd.yaml", ctrl.Log, true); err != nil {
			ctrl.Log.Error(err, "failed to apply etcd pod")
			return ctrl.Result{}, err
		}
		if _, err := etcdApp.Apply(ctx, "etcd/service.yaml", ctrl.Log, true); err != nil {
			ctrl.Log.Error(err, "failed to apply etcd service")
			return ctrl.Result{}, err
		}
	}

	if err := vineyarddApp.ApplyAll(ctx, vineyarddFile, ctrl.Log); err != nil {
		ctrl.Log.Error(err, "failed to apply vineyardd resources")
		return ctrl.Result{}, err
	}

	if err := r.UpdateStatus(ctx, &vineyardd); err != nil {
		ctrl.Log.Error(err, "failed to update status")
		return ctrl.Result{}, err
	}

	// reconcile every minute
	var duration, _ = time.ParseDuration("1m")
	return ctrl.Result{RequeueAfter: duration}, nil
}

// UpdateStatus updates the status of the Vineyardd.
func (r *VineyarddReconciler) UpdateStatus(ctx context.Context, vineyardd *k8sv1alpha1.Vineyardd) error {
	deployment := appsv1.Deployment{}
	if err := r.Client.Get(ctx, client.ObjectKey{Name: vineyardd.Name, Namespace: vineyardd.Namespace}, &deployment); err != nil {
		ctrl.Log.V(1).Error(err, "failed to get deployment")
	}

	// get the running vineyardd
	status := &k8sv1alpha1.VineyarddStatus{
		Running:    deployment.Status.ReadyReplicas,
		Required:   int32(vineyardd.Spec.Replicas),
		Conditions: deployment.Status.Conditions,
	}
	if err := r.updateStatus(ctx, vineyardd, status); err != nil {
		return fmt.Errorf("failed to update status: %w", err)
	}
	return nil
}

func (r *VineyarddReconciler) updateStatus(ctx context.Context, vineyardd *k8sv1alpha1.Vineyardd, status *k8sv1alpha1.VineyarddStatus) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		if err := r.Client.Get(ctx, client.ObjectKey{Name: vineyardd.Name, Namespace: vineyardd.Namespace}, vineyardd); err != nil {
			return fmt.Errorf("failed to get vineyardd: %w", err)
		}
		vineyardd.Status = *status
		vineyardd.Kind = "Vineyardd"
		if err := kubernetes.ApplyOverlay(vineyardd, &k8sv1alpha1.Vineyardd{Status: *status}); err != nil {
			return fmt.Errorf("failed to overlay vineyardd's status: %w", err)
		}
		if err := r.Client.Status().Update(ctx, vineyardd); err != nil {
			return fmt.Errorf("failed to update vineyardd's status: %w", err)
		}
		return nil
	})
}

// SetupWithManager sets up the controller with the Manager.
func (r *VineyarddReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.Vineyardd{}).
		Complete(r)
}
