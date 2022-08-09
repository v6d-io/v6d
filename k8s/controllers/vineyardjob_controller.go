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

package controllers

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/core"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	record "k8s.io/client-go/tools/record"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"

	v1alpha1 "github.com/v6d-io/v6d/k8s/api/k8s/v1alpha1"
	schedulers "github.com/v6d-io/v6d/k8s/schedulers"
)

// VineyardJobReconciler reconciles a VineyardJob object
type VineyardJobReconciler struct {
	client.Client
	Log       logr.Logger
	Scheme    *runtime.Scheme
	Recorder  record.EventRecorder
	Scheduler *schedulers.VineyardScheduler
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardjobs/status,verbs=get;update;patch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.

func (r *VineyardJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("vineyardjob", req.NamespacedName)
	log.Info("start reconcilering ...")

	var job v1alpha1.VineyardJob
	if err := r.Get(ctx, req.NamespacedName, &job); err != nil {
		// been deleted
		log.Info("vineyardjob been deleted")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	target := types.NamespacedName{Namespace: job.Namespace, Name: job.Name}
	return r.reconcilePod(ctx, &job, target)
}

func (r *VineyardJobReconciler) reconcilePod(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName) (ctrl.Result, error) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	log.Info("start reconcilering pod ...")
	r.Recorder.Event(job, core.EventTypeNormal, "Updated", "start reconcilering pods ...")

	if job.Spec.Replicas < 1 {
		return ctrl.Result{}, fmt.Errorf("Invalid replica value for pod: %d", job.Spec.Replicas)
	}

	var pod corev1.Pod
	if err := r.Get(ctx, types.NamespacedName{
		Namespace: target.Namespace,
		Name:      target.Name + "-0",
	}, &pod); err == nil {
		log.V(10).Info("pod already exists, it might be a status update request")
		if !r.ifPodReady(&pod) {
			go r.checkPodStatus(ctx, job, target, true)
		}
		return ctrl.Result{}, nil
	}

	log.Info("start creating pod for vineyardjob ...")
	r.Recorder.Event(job, core.EventTypeNormal, "Created", "start creating pods ...")

	placements, splits, err := r.Scheduler.ComputePlacementFor(ctx, job)
	if err != nil {
		placements = make([]string, job.Spec.Replicas)
		splits = make([][]string, job.Spec.Replicas)

		nodes, err := r.Scheduler.GetAllNodes(ctx)
		if err != nil {
			return ctrl.Result{}, err
		}

		for index := 0; index < job.Spec.Replicas; index++ {
			placements[index] = nodes[index%len(nodes)]
		}

		log.Info("failed to compute placement", "target", target, "error", err)
		// return err
	}

	applySchedulingHint := func(pod *corev1.Pod, rank int) error {
		if rank >= len(placements) || rank >= len(splits) {
			return fmt.Errorf("rank is out of range, rank is %d, placements are: %v, splits are %v", rank, placements, splits)
		}
		// patch env
		for index := range pod.Spec.Containers {
			container := &pod.Spec.Containers[index]
			container.Env = append(container.Env,
				corev1.EnvVar{Name: "k8s.v6d.io.placement", Value: placements[rank]},
				corev1.EnvVar{Name: "k8s.v6d.io.splits", Value: fmt.Sprint(splits[rank])})
		}
		pod.Annotations["k8s.v6d.io/placements"] = fmt.Sprint(placements)
		pod.Annotations["k8s.v6d.io/splits"] = fmt.Sprint(splits[rank])
		pod.Spec.NodeName = placements[rank]
		return nil
	}

	makePodForJob := func(job *v1alpha1.VineyardJob, rank int) (*corev1.Pod, error) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				Name:        target.Name + "-" + strconv.Itoa(rank),
				Namespace:   target.Namespace,
			},
			Spec: *job.Spec.Template.Spec.DeepCopy(),
		}
		for k, v := range job.Spec.Template.Annotations {
			pod.Annotations[k] = v
		}
		for k, v := range job.Annotations {
			pod.Annotations[k] = v
		}
		pod.Annotations["k8s.v6d.io/vineyardjob"] = job.Name
		for k, v := range job.Spec.Template.Labels {
			pod.Labels[k] = v
		}
		for k, v := range job.Labels {
			pod.Labels[k] = v
		}
		pod.Labels["k8s.v6d.io/vineyardjob"] = job.Name
		if err := applySchedulingHint(pod, rank); err != nil {
			log.Error(err, "failed to apply the scheduling hint")
			return nil, err
		}
		if err := ctrl.SetControllerReference(job, pod, r.Scheme); err != nil {
			log.Error(err, "failed to attach controller reference to pod")
			return nil, err
		}
		return pod, nil
	}

	for rank := 0; rank < job.Spec.Replicas; rank++ {
		podspec, err := makePodForJob(job, rank)
		if err != nil {
			log.Error(err, "failed to construct pod for job", "rank", rank, "spec", job.Spec.Template.Spec)
			return ctrl.Result{Requeue: false}, err
		}
		log.V(10).Info("pod spec", "pod", podspec)
		r.Recorder.Eventf(job, core.EventTypeNormal, "Created", "start creating pod %s ...", podspec.Name)
		if err := r.Create(ctx, podspec); err != nil {
			log.Error(err, "failed to create pod for job", "rank", rank)
			return ctrl.Result{Requeue: false}, err
		}
	}

	// check status
	go r.checkPodStatus(ctx, job, target, false)
	return ctrl.Result{}, nil
}

func (r *VineyardJobReconciler) ifPodReady(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodRunning
}

func (r *VineyardJobReconciler) checkPodStatus(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName, delay bool) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	r.waitForRefresh(delay)
	log.Info("checking pod status ...")

	job.Status.Replicas = job.Spec.Replicas
	job.Status.Ready = 0
	job.Status.Hosts = make([]string, job.Spec.Replicas)

	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(target.Namespace), client.MatchingLabels(map[string]string{"k8s.v6d.io/vineyardjob": job.Name})); err != nil {
		log.Info("failed to list pods", "target", target)
	}
	for i, pod := range pods.Items {
		job.Status.Hosts[i] = pod.Spec.NodeName
		if r.ifPodReady(&pod) {
			job.Status.Ready += 1
		}
	}
	r.Recorder.Eventf(job, core.EventTypeNormal, "Updated", "%d pods for job finished, expect %d ...", job.Status.Ready, job.Status.Replicas)
	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "failed to update the job status")
	}
}

func (r *VineyardJobReconciler) waitForRefresh(delay bool) {
	// wait for 60 seconds to refresh
	if delay {
		time.Sleep(time.Second * 60)
	} else {
		// make sure workload has been created
		time.Sleep(time.Second * 10)
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *VineyardJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.VineyardJob{}).
		Complete(r)
}
