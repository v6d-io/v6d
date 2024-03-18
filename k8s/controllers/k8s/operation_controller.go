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

// Package k8s contains controllers for k8s API group k8s.v6d.io
package k8s

import (
	"context"
	"time"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/operation"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// OperationReconciler reconciles a Operation object
type OperationReconciler struct {
	client.Client
	*kubernetes.Clientset
	record.EventRecorder
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=operations,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=operations/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=operations/finalizers,verbs=update
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;create;watch;update;list;delete
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get;create;update;list;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;create;update;delete
// +kubebuilder:rbac:groups="",resources=pods/status,verbs=get;list;create;update;delete
// +kubebuilder:rbac:groups="",resources=pods/log,verbs=get
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds,verbs=get;list;watch;create;update;patch;delete

// Reconcile reconciles the operation
func (r *OperationReconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithName("controllers").WithName("Operation")

	op := k8sv1alpha1.Operation{}
	if err := r.Get(ctx, req.NamespacedName, &op); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	logger.Info("Reconciling Operation", "operation", op)

	app := swckkube.Application{
		Client:   r.Client,
		FileRepo: templates.Repo,
		CR:       &op,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Operation"),
		TmplFunc: map[string]interface{}{
			"getDistributedAssemblyConfig": operation.GetDistributedAssemblyConfig,
			"getAssemblyConfig":            operation.GetAssemblyConfig,
			"getDaskRepartitionConfig":     operation.GetDaskRepartitionConfig,
		},
		Recorder: r.EventRecorder,
	}

	preOp := operation.NewPluggableOperation(op.Spec.Name, r.Client, r.Clientset, &app)
	if err := preOp.CreateJob(ctx, &op); err != nil {
		logger.Error(err, "Failed to create the job", "Operation", op)
		return ctrl.Result{}, err
	}

	if err := r.UpdateStatus(ctx, &op, preOp.IsDone()); err != nil {
		logger.Error(err, "Failed to update the status", "Operation", op)
	}

	// reconcile every minute
	return ctrl.Result{RequeueAfter: time.Minute}, nil
}

// UpdateStatus updates the status of the localobject
func (r *OperationReconciler) UpdateStatus(
	ctx context.Context,
	op *v1alpha1.Operation,
	opDone bool,
) error {
	state := v1alpha1.OperationRunning
	if opDone {
		state = v1alpha1.OperationSucceeded
	}

	status := &v1alpha1.OperationStatus{
		State: state,
	}
	if err := ApplyStatueUpdate(ctx, r.Client, op, r.Status(),
		func(op *v1alpha1.Operation) (error, *v1alpha1.Operation) {
			op.Status = *status
			op.Kind = "Operation"

			if err := swckkube.ApplyOverlay(op, &v1alpha1.Operation{Status: *status}); err != nil {
				return errors.Wrap(err, "failed to overlay operation's status"), nil
			}
			return nil, op
		},
	); err != nil {
		return errors.Wrap(err, "failed to update status")
	}

	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *OperationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.Operation{}).
		Complete(r)
}
