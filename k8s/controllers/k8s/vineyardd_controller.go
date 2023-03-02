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

// Package k8s contains all controllers in the vineyard operator.
package k8s

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// VineyarddReconciler reconciles a Vineyardd object
type VineyarddReconciler struct {
	client.Client
	record.EventRecorder
	Scheme *runtime.Scheme
}

// EtcdConfig holds all configuration about etcd
type EtcdConfig struct {
	Namespace string
	Rank      int
	Endpoints string
	Image     string
}

// Etcd contains the configuration about etcd
var Etcd EtcdConfig

// GetEtcdConfig get etcd configuration from Etcd
func getEtcdConfig() EtcdConfig {
	return Etcd
}

func getStorage(q resource.Quantity) string {
	return q.String()
}

// ServiceLabelSelector represents the label selector of the service
type ServiceLabelSelector struct {
	Key   string
	Value string
}

// SvcLabelSelector is the label selector of the service
var SvcLabelSelector []ServiceLabelSelector

func getServiceLabelSelector() []ServiceLabelSelector {
	return SvcLabelSelector
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
func (r *VineyarddReconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithName("controllers").WithName("Vineyardd")

	vineyardd := k8sv1alpha1.Vineyardd{}
	if err := r.Get(ctx, req.NamespacedName, &vineyardd); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	logger.V(1).Info("Reconciling Vineyardd", "vineyardd", vineyardd)

	vineyarddFile, err := templates.GetFilesRecursive("vineyardd")
	if err != nil {
		logger.Error(err, "failed to load vineyardd templates")
		return ctrl.Result{}, err
	}

	// preprocessing the socket directory
	k8sv1alpha1.PreprocessVineyarddSocket(&vineyardd)
	logger.Info("Rendered Vineyardd", "vineyardd", vineyardd)

	// deploy the vineyardd
	vineyarddApp := kubernetes.Application{
		Client:   r.Client,
		FileRepo: templates.Repo,
		CR:       &vineyardd,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Vineyardd"),
		Recorder: r.EventRecorder,
		TmplFunc: map[string]interface{}{
			"getStorage":              getStorage,
			"getServiceLabelSelector": getServiceLabelSelector,
		},
	}
	etcdApp := kubernetes.Application{
		Client:   r.Client,
		FileRepo: templates.Repo,
		CR:       &vineyardd,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Vineyardd"),
		TmplFunc: map[string]interface{}{"getEtcdConfig": getEtcdConfig},
	}
	// set up the etcd
	Etcd.Namespace = vineyardd.Namespace
	etcdEndpoints := make([]string, 0, vineyardd.Spec.Etcd.Replicas)
	replicas := vineyardd.Spec.Etcd.Replicas
	for i := 0; i < replicas; i++ {
		etcdEndpoints = append(
			etcdEndpoints,
			fmt.Sprintf("etcd%v=http://etcd%v:2380", strconv.Itoa(i), strconv.Itoa(i)),
		)
	}
	Etcd.Endpoints = strings.Join(etcdEndpoints, ",")
	// the etcd is built in the vineyardd image
	Etcd.Image = vineyardd.Spec.VineyardConfig.Image

	for i := 0; i < replicas; i++ {
		Etcd.Rank = i
		if _, err := etcdApp.Apply(ctx, "etcd/etcd.yaml", logger, true); err != nil {
			logger.Error(err, "failed to apply etcd pod")
			return ctrl.Result{}, err
		}
		if _, err := etcdApp.Apply(ctx, "etcd/service.yaml", logger, true); err != nil {
			logger.Error(err, "failed to apply etcd service")
			return ctrl.Result{}, err
		}
	}

	SvcLabelSelector = make([]ServiceLabelSelector, 1)
	SvcLabelSelector[0].Key = "app.v6d.io/service"
	SvcLabelSelector[0].Value = "vineyardd-rpc"
	if err := vineyarddApp.ApplyAll(ctx, vineyarddFile, logger); err != nil {
		logger.Error(err, "failed to apply vineyardd resources")
		return ctrl.Result{}, err
	}

	if err := r.UpdateStatus(ctx, &vineyardd); err != nil {
		logger.Error(err, "failed to update status")
		return ctrl.Result{}, err
	}

	// reconcile every minute
	duration, _ := time.ParseDuration("1m")
	return ctrl.Result{RequeueAfter: duration}, nil
}

// UpdateStatus updates the status of the Vineyardd.
func (r *VineyarddReconciler) UpdateStatus(
	ctx context.Context,
	vineyardd *k8sv1alpha1.Vineyardd,
) error {
	name := client.ObjectKey{Name: vineyardd.Name, Namespace: vineyardd.Namespace}
	deployment := appsv1.Deployment{}
	if err := r.Get(ctx, name, &deployment); err != nil {
		log.V(1).Error(err, "failed to get deployment")
	}

	// get the running vineyardd
	status := &k8sv1alpha1.VineyarddStatus{
		ReadyReplicas: deployment.Status.ReadyReplicas,
		Conditions:    deployment.Status.Conditions,
	}
	if err := r.applyStatusUpdate(ctx, vineyardd, status); err != nil {
		return errors.Wrap(err, "failed to update status")
	}
	return nil
}

func (r *VineyarddReconciler) applyStatusUpdate(ctx context.Context,
	vineyardd *k8sv1alpha1.Vineyardd, status *k8sv1alpha1.VineyarddStatus,
) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		name := client.ObjectKey{Name: vineyardd.Name, Namespace: vineyardd.Namespace}
		if err := r.Get(ctx, name, vineyardd); err != nil {
			return errors.Wrap(err, "failed to get vineyardd")
		}
		vineyardd.Status = *status
		vineyardd.Kind = "Vineyardd"
		if err := kubernetes.ApplyOverlay(vineyardd, &k8sv1alpha1.Vineyardd{Status: *status}); err != nil {
			return errors.Wrap(err, "failed to overlay vineyardd's status")
		}
		if err := r.Status().Update(ctx, vineyardd); err != nil {
			return errors.Wrap(err, "failed to update vineyardd's status")
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
