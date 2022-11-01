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

// Package k8s contains controllers for k8s API group k8s.v6d.io
package k8s

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"
	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/operation"
)

// OperationReconciler reconciles a Operation object
type OperationReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Template kubernetes.Repo
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=operations,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=operations/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=operations/finalizers,verbs=update
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=pods/status,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=pods/log,verbs=get
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete

// Reconcile reconciles the operation
func (r *OperationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)
	op := k8sv1alpha1.Operation{}
	if err := r.Client.Get(ctx, req.NamespacedName, &op); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	ctrl.Log.V(1).Info("Reconciling Operation", "vineyardd", op)

	app := kubernetes.Application{
		Client:   r.Client,
		FileRepo: r.Template,
		CR:       &op,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Operation"),
		TmplFunc: map[string]interface{}{"getDistributedAssemblyConfig": operation.GetDistributedAssemblyConfig,
			"getAssemblyConfig": operation.GetAssemblyConfig, "getDaskRepartitionConfig": operation.GetDaskRepartitionConfig},
		Recorder: r.Recorder,
	}

	preOp := operation.NewPluggableOperation(op.Spec.Name, r.Client, &app)
	if err := preOp.CreateJob(ctx, &op); err != nil {
		ctrl.Log.Error(err, "Failed to create the job", "Operation", op)
		return ctrl.Result{}, err
	}

	if err := r.UpdateStatus(ctx, &op, preOp.IsDone()); err != nil {
		ctrl.Log.Error(err, "Failed to update the status", "Operation", op)
	}

	// reconcile every minute
	var duration, _ = time.ParseDuration("1m")
	return ctrl.Result{RequeueAfter: duration}, nil
}

// UpdateStatus updates the status of the localobject
func (r *OperationReconciler) UpdateStatus(ctx context.Context, op *v1alpha1.Operation, opDone bool) error {
	state := "running"
	if opDone {
		state = operation.SucceededState
	}

	status := &v1alpha1.OperationStatus{
		State: state,
	}
	if err := r.updateStatus(ctx, op, status); err != nil {
		return fmt.Errorf("failed to update status: %w", err)
	}
	return nil
}

func (r *OperationReconciler) updateStatus(ctx context.Context, op *v1alpha1.Operation, status *v1alpha1.OperationStatus) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		if err := r.Client.Get(ctx, client.ObjectKey{Name: op.Name, Namespace: op.Namespace}, op); err != nil {
			return fmt.Errorf("failed to get operation: %w", err)
		}
		op.Status = *status
		op.Kind = "Operation"

		if err := kubernetes.ApplyOverlay(op, &v1alpha1.Operation{Status: *status}); err != nil {
			return fmt.Errorf("failed to overlay operation's status: %w", err)
		}
		if err := r.Client.Status().Update(ctx, op); err != nil {
			return fmt.Errorf("failed to update operation's status: %w", err)
		}
		return nil
	})
}

// SetupWithManager sets up the controller with the Manager.
func (r *OperationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.Operation{}).
		Complete(r)
}
