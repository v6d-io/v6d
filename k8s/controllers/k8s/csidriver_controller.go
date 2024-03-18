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

package k8s

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"
	"github.com/pkg/errors"

	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
)

// CSIDriverReconciler reconciles a CSIDriver object
type CSIDriverReconciler struct {
	client.Client
	record.EventRecorder
	Scheme *runtime.Scheme
}

type StorageConfig struct {
	Namespace         string
	Name              string
	VolumeBindingMode string
}

//+kubebuilder:rbac:groups=k8s.v6d.io,resources=csidrivers,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=k8s.v6d.io,resources=csidrivers/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=k8s.v6d.io,resources=csidrivers/finalizers,verbs=update
//+kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update
//+kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch;create;update
//+kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
//+kubebuilder:rbac:groups="",resources=persistentvolumes,verbs=get;list;watch;update;patch;create;delete
//+kubebuilder:rbac:groups="",resources=persistentvolumes/finalizers,verbs=patch
//+kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch;update
//+kubebuilder:rbac:groups="",resources=persistentvolumeclaims/finalizers,verbs=patch
//+kubebuilder:rbac:groups=storage.k8s.io, resources=storageclasses, verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=storage.k8s.io, resources=csinodes, verbs=get;list;watch
//+kubebuilder:rbac:groups=storage.k8s.io,resources=volumeattachments,verbs=get;list;watch;update;patch
//+kubebuilder:rbac:groups=storage.k8s.io,resources=volumeattachments/status,verbs=get;list;watch;update;patch
//+kubebuilder:rbac:groups=csi.storage.k8s.io,resources=csinodeinfos,verbs=get;list;watch
//+kubebuilder:rbac:groups=snapshot.storage.k8s.io,resources=volumesnapshotclasses,verbs=get;list;watch
//+kubebuilder:rbac:groups=snapshot.storage.k8s.io,resources=volumesnapshotcontents,verbs=get;list;watch;create;update;delete
//+kubebuilder:rbac:groups=snapshot.storage.k8s.io,resources=volumesnapshots,verbs=get;list;watch;update

func (r *CSIDriverReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithName("controllers").WithName("CSIDriver")

	csiDriver := &k8sv1alpha1.CSIDriver{}
	if err := r.Client.Get(ctx, req.NamespacedName, csiDriver); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	logger.Info("Reconciling CSIDriver", "csiDriver", csiDriver)

	// check there is no more than one csi driver in the cluster
	csidrivers := &k8sv1alpha1.CSIDriverList{}
	if err := r.Client.List(ctx, csidrivers); err != nil {
		return ctrl.Result{}, err
	}

	if len(csidrivers.Items) > 1 {
		logger.Error(nil, "There is already a csi driver in the cluster")
		return ctrl.Result{}, nil
	}

	// check the vineyard clusters are ready
	if len(csiDriver.Spec.Clusters) == 0 {
		logger.Error(nil, "No vineyard cluster is specified")
		return ctrl.Result{}, nil
	}
	deployment := &appsv1.Deployment{}
	for _, c := range csiDriver.Spec.Clusters {
		if err := r.Client.Get(ctx, types.NamespacedName{Namespace: c.Namespace, Name: c.Name}, deployment); err != nil {
			return ctrl.Result{}, err
		}
		if deployment.Status.ReadyReplicas != *deployment.Spec.Replicas {
			logger.Error(nil, fmt.Sprintf("Vineyard deployment %s/%s is not ready", c.Namespace, c.Name))
			return ctrl.Result{}, nil
		}
	}

	// get the namespace of the vineyard operator
	deploymentLists := &appsv1.DeploymentList{}
	if err := r.Client.List(ctx, deploymentLists, client.MatchingLabels{"k8s.v6d.io/instance": "vineyard-operator"}); err != nil {
		return ctrl.Result{}, err
	}
	if len(deploymentLists.Items) != 1 {
		log.Errorf(nil, "Only one vineyard operator is allowed in the specific namespace, but got %v", len(deploymentLists.Items))
		return ctrl.Result{}, nil
	}
	if deploymentLists.Items[0].Status.ReadyReplicas != *deploymentLists.Items[0].Spec.Replicas {
		log.Errorf(nil, "Vineyard operator is not ready")
		return ctrl.Result{}, nil
	}

	// the csi driver should be in the same namespace as the vineyard operator
	// for sharing the same service account
	csiDriver.Namespace = deploymentLists.Items[0].Namespace
	// create a csi driver
	csiDriverApp := kubernetes.Application{
		Client:   r.Client,
		CR:       csiDriver,
		FileRepo: templates.Repo,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("CSIDriver"),
		Recorder: r.EventRecorder,
		TmplFunc: map[string]interface{}{},
	}
	if _, err := csiDriverApp.Apply(ctx, "csidriver/daemonset.yaml", logger, true); err != nil {
		logger.Error(err, "failed to apply csidriver daemonset manifest")
		return ctrl.Result{}, err
	}
	if _, err := csiDriverApp.Apply(ctx, "csidriver/deployment.yaml", logger, true); err != nil {
		logger.Error(err, "failed to apply csidriver deployment manifest")
		return ctrl.Result{}, err
	}
	for i := 0; i < len(csiDriver.Spec.Clusters); i++ {
		csiDriverApp.TmplFunc["getStorageConfig"] = func() StorageConfig {
			return StorageConfig{
				Namespace:         csiDriver.Spec.Clusters[i].Namespace,
				Name:              csiDriver.Spec.Clusters[i].Name,
				VolumeBindingMode: csiDriver.Spec.VolumeBindingMode,
			}
		}

		if _, err := csiDriverApp.Apply(ctx, "csidriver/storageclass.yaml", logger, true); err != nil {
			logger.Error(err, "failed to apply csidriver manifests")
			return ctrl.Result{}, err
		}
	}

	if err := r.UpdateStatus(ctx, csiDriver); err != nil {
		logger.Error(err, "Failed to update the status", "CSIDriver", csiDriver)
	}
	// reconcile every minute
	return ctrl.Result{RequeueAfter: time.Minute}, nil
}

func (r *CSIDriverReconciler) UpdateStatus(ctx context.Context, csiDriver *k8sv1alpha1.CSIDriver) error {
	depOK := false
	daeOK := false
	depName := csiDriver.Name + "-csi-driver"
	ns := csiDriver.Namespace
	// check if the csi driver deployment is ready
	dep := &appsv1.Deployment{}
	if err := r.Client.Get(ctx, types.NamespacedName{Namespace: ns, Name: depName}, dep); err != nil {
		return errors.Wrap(err, "failed to get csi driver deployment")
	}
	if dep.Status.ReadyReplicas == *dep.Spec.Replicas {
		depOK = true
	}

	// check if the csi nodes daemonset is ready
	daeName := csiDriver.Name + "-csi-nodes"
	dae := &appsv1.DaemonSet{}
	if err := r.Client.Get(ctx, types.NamespacedName{Namespace: ns, Name: daeName}, dae); err != nil {
		return errors.Wrap(err, "failed to get csi driver daemonset")
	}
	if dae.Status.NumberReady == dae.Status.DesiredNumberScheduled {
		daeOK = true
	}

	state := k8sv1alpha1.CSIDriverRunning
	if !depOK || !daeOK {
		state = k8sv1alpha1.CSIDriverPending
	}

	status := &k8sv1alpha1.CSIDriverStatus{
		State: state,
	}
	if err := ApplyStatueUpdate(ctx, r.Client, csiDriver, r.Status(),
		func(c *k8sv1alpha1.CSIDriver) (error, *k8sv1alpha1.CSIDriver) {
			csiDriver.Status = *status
			csiDriver.Kind = "CSIDriver"

			if err := kubernetes.ApplyOverlay(csiDriver, &v1alpha1.CSIDriver{Status: *status}); err != nil {
				return errors.Wrap(err, "failed to overlay csidriver's status"), nil
			}
			return nil, csiDriver
		},
	); err != nil {
		return errors.Wrap(err, "failed to update status")
	}

	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *CSIDriverReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.CSIDriver{}).
		Complete(r)
}
