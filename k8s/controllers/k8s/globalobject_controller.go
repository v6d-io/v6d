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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/schedulers"
)

const (
	// MaxDuration replesents the max duration of waiting the distributed assembly globalobject
	MaxDuration = 2 * time.Minute
)

// GlobalObjectReconciler reconciles a GlobalObject object
type GlobalObjectReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Template kubernetes.Repo
	Recorder record.EventRecorder
}

// DistributedAssemblyConfig is the config for the distributed assembly job
type DistributedAssemblyConfig struct {
	Name                 string
	Namespace            string
	StreamID             string
	JobName              string
	GLOBALOBJECT_ID      string
	OldObjectToNewObject string
	VineyardSockPath     string
}

// GlobalDistributedAssemblyConfig is the global config for the assembly job
var GlobalDistributedAssemblyConfig DistributedAssemblyConfig

func getDistributedAssemblyConfig() DistributedAssemblyConfig {
	return GlobalDistributedAssemblyConfig
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=globalobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=localobjects/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch

func (r *GlobalObjectReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = context.Background()
	// reconcile every minute
	var duration, _ = time.ParseDuration("1m")

	ctrl.Log.V(1).Info("Reconciling LocalObject...")

	job := batchv1.Job{}

	app := kubernetes.Application{
		Client:   r.Client,
		FileRepo: r.Template,
		CR:       &job,
		GVK:      batchv1.SchemeGroupVersion.WithKind("Job"),
		Recorder: r.Recorder,
		TmplFunc: map[string]interface{}{"getDistributedAssemblyConfig": getDistributedAssemblyConfig},
	}

	globalobjectList := &v1alpha1.GlobalObjectList{}
	if err := r.List(ctx, globalobjectList); err != nil {
		ctrl.Log.Error(err, "unable to list globalobjects")
		return ctrl.Result{}, err
	}

	// find the global object which need to assemble the distributed object
	for _, globalobject := range globalobjectList.Items {
		pod, err := r.FindDistributedAssemblyPodByGlobalObject(ctx, &globalobject)
		// if the pod is not found, it means there is error when reconcile the globalobject
		if err != nil {
			ctrl.Log.Error(err, "failed to find the pod which need to be injected with the assembly job")
			return ctrl.Result{}, err
		}
		if pod != nil {
			// create the distibuted assembly pod
			if r.NeedDistribuedAssemblyJob(ctx, &globalobject, pod) {
				if _, err := app.Apply(ctx, "assembly/distributed-assembly-job.yaml", ctrl.Log, false); err != nil {
					ctrl.Log.Error(err, "failed to apply distributed assembly job")
					return ctrl.Result{}, err
				}
				if err := r.UpdateState(ctx, &globalobject, FailedState, pod.Namespace); err != nil {
					ctrl.Log.Error(err, "failed to update the assembly globalobject's state")
					return ctrl.Result{}, err
				}
			} else {
				if err := r.UpdateTime(ctx, &globalobject); err != nil {
					ctrl.Log.Error(err, "failed to update the common globalobject's time")
					return ctrl.Result{}, err
				}
				// If the typename contains the stream, we will reconcile it into the next duration
				if strings.Contains(strings.ToLower(globalobject.Spec.Typename), "stream") && MaxDuration > time.Since(globalobject.Status.CreationTime.Time) {
					return ctrl.Result{RequeueAfter: duration}, nil
				}
				if err := r.UpdateState(ctx, &globalobject, SucceededState, pod.Namespace); err != nil {
					ctrl.Log.Error(err, "failed to update the common globalobject's state")
					return ctrl.Result{}, err
				}
			}
		}
	}

	return ctrl.Result{RequeueAfter: duration}, nil
}

// FindDistributedAssemblyPodByGlobalObject finds the pod which need to be injected with the assembly job
func (r *GlobalObjectReconciler) FindDistributedAssemblyPodByGlobalObject(ctx context.Context, globalObject *v1alpha1.GlobalObject) (*corev1.Pod, error) {
	labels := globalObject.Labels
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

func (r *GlobalObjectReconciler) NeedDistribuedAssemblyJob(ctx context.Context, globalObject *v1alpha1.GlobalObject, pod *corev1.Pod) bool {
	podLabels := pod.Labels
	signatures := map[string]bool{}
	for i := range globalObject.Spec.Members {
		signatures[globalObject.Spec.Members[i]] = true
	}

	localobjectList := &v1alpha1.LocalObjectList{}
	if err := r.List(ctx, localobjectList); err != nil {
		ctrl.Log.Error(err, "unable to list localobjects")
		return false
	}

	sigToId := map[string]string{}
	for i := range localobjectList.Items {
		if _, ok := signatures[localobjectList.Items[i].Spec.Signature]; ok &&
			strings.Contains(strings.ToLower(localobjectList.Items[i].Spec.Typename), "stream") {
			sigToId[localobjectList.Items[i].Spec.Signature] = localobjectList.Items[i].Name
		}
	}

	globalobjectList := &v1alpha1.GlobalObjectList{}
	if err := r.List(ctx, globalobjectList); err != nil {
		ctrl.Log.Error(err, "unable to list globalobjects")
		return false
	}

	oldObjectToNewObject := map[string]string{}
	for i := range globalobjectList.Items {
		labels := globalobjectList.Items[i].Labels
		if v, ok := labels[PodNameLabelKey]; ok {
			for j := range sigToId {
				if strings.Contains(v, sigToId[j]) {
					oldObjectToNewObject[sigToId[j]] = globalobjectList.Items[i].Name
				}
			}
		}
	}

	// Apply the distributed assembly job
	if len(sigToId) == len(oldObjectToNewObject) {
		str := `'{`
		for k, v := range oldObjectToNewObject {
			str = str + `"` + k + `"` + ":" + `"` + v + `"` + ","
		}
		str = str[:len(str)-1] + `}'`
		GlobalDistributedAssemblyConfig.Name = AssemblyPrefix + globalObject.Name
		GlobalDistributedAssemblyConfig.Namespace = pod.Namespace
		GlobalDistributedAssemblyConfig.GLOBALOBJECT_ID = globalObject.Name
		GlobalDistributedAssemblyConfig.OldObjectToNewObject = str
		GlobalDistributedAssemblyConfig.JobName = podLabels[schedulers.VineyardJobName]
		vineyardd := podLabels[schedulers.VineyarddName]
		GlobalDistributedAssemblyConfig.VineyardSockPath = "/var/run/vineyard-" + globalObject.Namespace + "-" + vineyardd
		return true
	}
	return false
}

func (r *GlobalObjectReconciler) UpdateState(ctx context.Context, globalobject *v1alpha1.GlobalObject, defaultValue string, namespace string) error {
	job := &batchv1.Job{}
	state := defaultValue
	err := r.Client.Get(ctx, client.ObjectKey{Name: AssemblyPrefix + globalobject.Spec.ObjectID, Namespace: namespace}, job)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	// if the job exist
	if !apierrors.IsNotFound(err) {
		if job.Status.Succeeded == 1 {
			state = SucceededState
			// get the new produced globalobject's ID by the distributed assembly job
			newGlobalObjectId := ""
			globalobjectList := &v1alpha1.GlobalObjectList{}
			if err := r.Client.List(ctx, globalobjectList); err != nil {
				ctrl.Log.Error(err, "unable to list globalobjects")
				return err
			}

			for i := range globalobjectList.Items {
				labels := globalobjectList.Items[i].Labels
				if v, ok := labels[PodNameLabelKey]; ok {
					if strings.Contains(v, job.Name) {
						newGlobalObjectId = globalobjectList.Items[i].Name
						break
					}
				}
			}

			// update the configmap to store the new produced globalobject's ID
			configmapName := globalobject.Labels["job"]
			configmapNamespace := namespace
			// update the configmap
			configmap := &corev1.ConfigMap{}
			if err := r.Client.Get(ctx, client.ObjectKey{Name: configmapName, Namespace: configmapNamespace}, configmap); err != nil {
				ctrl.Log.Info("failed to get the configmap")
			}
			if !apierrors.IsNotFound(err) {
				cm := corev1.ConfigMap{
					TypeMeta: metav1.TypeMeta{
						Kind:       "ConfigMap",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      configmapName,
						Namespace: configmapNamespace,
					},
					Data: map[string]string{configmapName: newGlobalObjectId},
				}
				if err := r.Client.Create(ctx, &cm); err != nil {
					ctrl.Log.Error(err, "failed to create the configmap")
					return err
				}
			} else {
				data := configmap.Data
				data[configmapName] = newGlobalObjectId
				if err := r.Client.Update(ctx, configmap); err != nil {
					ctrl.Log.Error(err, "failed to update the configmap")
					return err
				}
			}
		}
	}

	status := &v1alpha1.GlobalObjectStatus{
		State: state,
	}
	if err := r.updateStatus(ctx, globalobject, status); err != nil {
		return err
	}

	return nil
}

func (r *GlobalObjectReconciler) UpdateTime(ctx context.Context, globalobject *v1alpha1.GlobalObject) error {
	nilTime := metav1.Time{}
	if globalobject.Status.CreationTime != nilTime {
		return nil
	}

	status := &v1alpha1.GlobalObjectStatus{
		CreationTime: metav1.Now(),
	}
	if err := r.updateStatus(ctx, globalobject, status); err != nil {
		return err
	}

	return nil
}

func (r *GlobalObjectReconciler) updateStatus(ctx context.Context, globalobject *v1alpha1.GlobalObject, status *v1alpha1.GlobalObjectStatus) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		if err := r.Client.Get(ctx, client.ObjectKey{Name: globalobject.Name, Namespace: globalobject.Namespace}, globalobject); err != nil {
			return fmt.Errorf("failed to get globalobject: %w", err)
		}
		globalobject.Status = *status
		globalobject.Kind = "GlobalObject"

		if err := kubernetes.ApplyOverlay(globalobject, &v1alpha1.GlobalObject{Status: *status}); err != nil {
			return fmt.Errorf("failed to overlay globalobject's status: %w", err)
		}
		if err := r.Client.Status().Update(ctx, globalobject); err != nil {
			return fmt.Errorf("failed to update globalobject's status: %w", err)
		}
		return nil
	})
}

func (r *GlobalObjectReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.GlobalObject{}).
		Complete(r)
}
