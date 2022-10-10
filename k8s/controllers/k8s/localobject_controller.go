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
	"strings"

	"time"

	"github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/operator"
	"github.com/v6d-io/v6d/k8s/schedulers"
)

// LocalObjectReconciler reconciles a LocalObject object
type LocalObjectReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Template kubernetes.Repo
	Recorder record.EventRecorder
}

// AssemblyConfig is the config for the assembly job
type AssemblyConfig struct {
	Name             string
	Namespace        string
	StreamID         string
	JobName          string
	NodeName         string
	VineyardSockPath string
}

const (
	// PodNameLabelKey is the label key for the pod name which generated the stream
	PodNameLabelKey = "created-by-podname"
	// PodNamespaceLabelKey is the label key for the pod namespace which generated the stream
	PodNameSpaceLabelKey = "created-by-podnamespace"
	// NeedInjectedAssemblyKey represents the pod need to be injected with the assembly job
	NeedInjectedAssemblyKey = "need-injected-assembly"
	// AssemblyPrefix is the prefix of the assembly job
	AssemblyPrefix = "assemble-"
	// SucceededState is the succeeded state of the object
	SucceededState = "succeeded"
	// FailedState is the failed state of the object
	FailedState = "failed"
)

// GlobalAssemblyConfig is the global config for the assembly job
var GlobalAssemblyConfig AssemblyConfig

func getAssemblyConfig() AssemblyConfig {
	return GlobalAssemblyConfig
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=pods/status,verbs=get;list;watch;create;update;patch

func (r *LocalObjectReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = context.Background()

	ctrl.Log.V(1).Info("Reconciling LocalObject...")

	job := batchv1.Job{}

	app := kubernetes.Application{
		Client:   r.Client,
		FileRepo: r.Template,
		CR:       &job,
		GVK:      batchv1.SchemeGroupVersion.WithKind("Job"),
		Recorder: r.Recorder,
		TmplFunc: map[string]interface{}{"getAssemblyConfig": getAssemblyConfig},
	}

	localObjectList := &v1alpha1.LocalObjectList{}
	if err := r.Client.List(ctx, localObjectList); err != nil {
		ctrl.Log.Error(err, "failed to list LocalObjects")
		return ctrl.Result{}, err
	}

	needDeleteLabel := false
	// find the local stream object which need to transform the stream into object and create the assembly job to consume it
	for _, localObject := range localObjectList.Items {
		pod, err := r.FindNeedAssemblyPodByLocalObject(ctx, &localObject)
		// if the pod is not found, it means there is error when reconcile the localobject
		if err != nil {
			ctrl.Log.Error(err, "failed to find the pod which need to be injected with the assembly job")
			return ctrl.Result{}, err
		}
		if pod != nil {
			// create the assembly pod in the same node with the pod which generated the stream
			if r.NeedAssemblyJob(ctx, &localObject, pod) {
				needDeleteLabel = true
				if _, err := app.Apply(ctx, "assembly/local-assembly-job.yaml", ctrl.Log, false); err != nil {
					ctrl.Log.Error(err, "failed to apply assembly job")
					return ctrl.Result{}, err
				}
				if err := r.UpdateStatus(ctx, &localObject, FailedState, pod.Namespace); err != nil {
					ctrl.Log.Error(err, "failed to update assembly localobject status")
					return ctrl.Result{}, err
				}
			} else {
				if err := r.UpdateStatus(ctx, &localObject, SucceededState, pod.Namespace); err != nil {
					ctrl.Log.Error(err, "failed to update common localobject status")
					return ctrl.Result{}, err
				}
			}
		}

	}

	if needDeleteLabel {
		if err := r.DeleteAssemblyEnabledLable(ctx); err != nil {
			ctrl.Log.Error(err, "failed to delete the assembly enabled label")
			return ctrl.Result{}, err
		}
	}

	// reconcile every minute
	var duration, _ = time.ParseDuration("1m")
	return ctrl.Result{RequeueAfter: duration}, nil
}

// DeleteAssemblyEnabledLable will delete the label when all localobjects are transformed into local chunk
func (r *LocalObjectReconciler) DeleteAssemblyEnabledLable(ctx context.Context) error {
	podList := &corev1.PodList{}
	opts := []client.ListOption{
		client.MatchingLabels{
			operator.AssmeblyEnabledLabel: "true",
		},
	}
	if err := r.Client.List(ctx, podList, opts...); err != nil {
		return err
	}

	// record all pods which needs the assembly job and their required jobs
	for i := range podList.Items {
		annotations := podList.Items[i].GetAnnotations()
		if requiredJob, ok := annotations[schedulers.VineyardJobRequired]; ok {
			jobs := strings.Split(requiredJob, ".")
			deleted := true
			for _, job := range jobs {
				// get the localobjects produced by these jobs
				localobjectList := &v1alpha1.LocalObjectList{}
				opts := []client.ListOption{
					client.MatchingLabels{
						"job": job,
					},
				}
				if err := r.Client.List(ctx, localobjectList, opts...); err != nil {
					ctrl.Log.Error(err, "failed to list localobjects")
					return err
				}
				for _, localObject := range localobjectList.Items {
					createdPod := localObject.Labels[PodNameLabelKey]
					if !strings.Contains(strings.ToLower(createdPod), AssemblyPrefix) && localObject.Status.State != SucceededState {
						deleted = false
						break
					}
				}
				if !deleted {
					break
				}
				// get the globalobjects produced by these jobs
				globalobjectList := &v1alpha1.GlobalObjectList{}
				if err := r.Client.List(ctx, globalobjectList, opts...); err != nil {
					ctrl.Log.Error(err, "failed to list globalobjects")
					return err
				}
				for _, globalObject := range globalobjectList.Items {
					createdPod := globalObject.Labels[PodNameLabelKey]
					if !strings.Contains(strings.ToLower(createdPod), AssemblyPrefix) && globalObject.Status.State != SucceededState {
						deleted = false
						break
					}
				}
				if !deleted {
					break
				}
			}
			// delete the label if all required localobjects and globalobjects are succeeded
			if deleted {
				delete(podList.Items[i].Labels, operator.AssmeblyEnabledLabel)
				if err := r.Client.Update(ctx, &podList.Items[i], &client.UpdateOptions{}); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// FindNeedAssemblyPodByLocalObject finds the pod which need to be injected with the assembly job
func (r *LocalObjectReconciler) FindNeedAssemblyPodByLocalObject(ctx context.Context, localObject *v1alpha1.LocalObject) (*corev1.Pod, error) {
	labels := localObject.Labels
	podName := labels[PodNameLabelKey]
	podNamespace := labels[PodNameSpaceLabelKey]
	if podName != "" && podNamespace != "" {
		pod := &corev1.Pod{}
		if err := r.Client.Get(ctx, client.ObjectKey{Name: podName, Namespace: podNamespace}, pod); err != nil {
			ctrl.Log.Error(err, "failed to get the pod")
			return nil, err
		}
		if v, ok := pod.Labels[NeedInjectedAssemblyKey]; ok && strings.ToLower(v) == "true" {
			return pod, nil
		}
	}
	return nil, nil
}

func (r *LocalObjectReconciler) NeedAssemblyJob(ctx context.Context, localObject *v1alpha1.LocalObject, pod *corev1.Pod) bool {
	podLabels := pod.Labels
	// When the pod which generated the stream is annotated, the assembly job will be created in the same pod
	if _, ok := podLabels[NeedInjectedAssemblyKey]; ok {
		if strings.Contains(strings.ToLower(localObject.Spec.Typename), "stream") {
			GlobalAssemblyConfig.Name = AssemblyPrefix + localObject.Spec.ObjectID
			GlobalAssemblyConfig.Namespace = pod.Namespace
			GlobalAssemblyConfig.StreamID = localObject.Spec.ObjectID
			GlobalAssemblyConfig.NodeName = localObject.Spec.Hostname
			GlobalAssemblyConfig.JobName = podLabels[schedulers.VineyardJobName]
			vineyardd := podLabels[schedulers.VineyarddName]
			GlobalAssemblyConfig.VineyardSockPath = "/var/run/vineyard-" + localObject.Namespace + "-" + vineyardd
			return true
		}
	}
	return false
}

func (r *LocalObjectReconciler) UpdateStatus(ctx context.Context, localobject *v1alpha1.LocalObject, defaultValue string, namespace string) error {
	job := &batchv1.Job{}
	state := defaultValue
	err := r.Client.Get(ctx, client.ObjectKey{Name: AssemblyPrefix + localobject.Spec.ObjectID, Namespace: namespace}, job)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	// if the job exist
	if !apierrors.IsNotFound(err) {
		if job.Status.Succeeded == 1 {
			state = SucceededState
		}
	}

	status := &v1alpha1.LocalObjectStatus{
		State: state,
	}
	if err := r.updateStatus(ctx, localobject, status); err != nil {
		return err
	}

	return nil
}

func (r *LocalObjectReconciler) updateStatus(ctx context.Context, localobject *v1alpha1.LocalObject, status *v1alpha1.LocalObjectStatus) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		if err := r.Client.Get(ctx, client.ObjectKey{Name: localobject.Name, Namespace: localobject.Namespace}, localobject); err != nil {
			return fmt.Errorf("failed to get localobject: %w", err)
		}
		localobject.Status = *status
		localobject.Kind = "LocalObject"

		if err := kubernetes.ApplyOverlay(localobject, &v1alpha1.LocalObject{Status: *status}); err != nil {
			return fmt.Errorf("failed to overlay localobject's status: %w", err)
		}
		if err := r.Client.Status().Update(ctx, localobject); err != nil {
			return fmt.Errorf("failed to update localobject's status: %w", err)
		}
		return nil
	})
}

func (r *LocalObjectReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.LocalObject{}).
		Complete(r)
}
