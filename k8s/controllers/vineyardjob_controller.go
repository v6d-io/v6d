/** Copyright 2020-2021 Alibaba Group Holding Limited.

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
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	appsv1 "k8s.io/api/apps/v1"
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
	Scheduler *schedulers.VineyardScheduler
}

//+kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardjobs/status,verbs=get;update;patch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the VineyardJob object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.6.4/pkg/reconcile
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
	if job.Spec.Kind == "pod" {
		return r.reconcilePod(ctx, &job, target)
	}
	if job.Spec.Kind == "deployment" {
		return r.reconcileDeployment(ctx, &job, target)
	}
	if job.Spec.Kind == "replicaset" {
		return r.reconcileReplicaSet(ctx, &job, target)
	}
	if job.Spec.Kind == "statefulset" {
		return r.reconcileStatefulSet(ctx, &job, target)
	}

	return ctrl.Result{}, fmt.Errorf("unknown workload kind: %s", job.Spec.Kind)
}

func (r *VineyardJobReconciler) reconcilePod(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName) (ctrl.Result, error) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	log.Info("start reconcilering pod ...")

	if job.Spec.Replica < 1 {
		return ctrl.Result{}, fmt.Errorf("Invalid replica value for pod: %d", job.Spec.Replica)
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

	placements, splits, err := r.Scheduler.ComputePlacementFor(ctx, job)
	if err != nil {
		placements = make([]string, job.Spec.Replica)
		splits = make([][]string, job.Spec.Replica)

		nodes, err := r.Scheduler.GetAllNodes(ctx)
		if err != nil {
			return ctrl.Result{}, err
		}

		for index := 0; index < job.Spec.Replica; index++ {
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
			Spec: *job.Spec.Pod.DeepCopy(),
		}
		for k, v := range job.Annotations {
			pod.Annotations[k] = v
		}
		pod.Annotations["k8s.v6d.io/vineyardjob"] = job.Name
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

	for rank := 0; rank < job.Spec.Replica; rank++ {
		podspec, err := makePodForJob(job, rank)
		if err != nil {
			log.Error(err, "failed to construct pod for job", "rank", rank, "spec", job.Spec.Pod)
			return ctrl.Result{Requeue: false}, err
		}
		log.Info("pod spec", "pod", podspec)
		if err := r.Create(ctx, podspec); err != nil {
			log.Error(err, "failed to create pod for job", "rank", rank)
			return ctrl.Result{Requeue: false}, err
		}
	}

	// check status
	go r.checkPodStatus(ctx, job, target, false)
	return ctrl.Result{}, nil
}

func (r *VineyardJobReconciler) reconcileDeployment(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName) (ctrl.Result, error) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	log.Info("start reconcilering deployment ...")

	var deployment appsv1.Deployment
	if err := r.Get(ctx, target, &deployment); err == nil {
		log.V(10).Info("deployment already exists, it might be a status update request")
		if !r.ifDeploymentReady(&deployment) {
			go r.checkDeploymentStatus(ctx, job, target, true)
		}
		return ctrl.Result{}, nil
	}

	log.Info("start creating deployment for vineyardjob ...")

	makeDeploymentForJob := func(job *v1alpha1.VineyardJob) (*appsv1.Deployment, error) {
		deployment := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				Name:        job.Name,
				Namespace:   job.Namespace,
			},
			Spec: *job.Spec.Deployment.DeepCopy(),
		}
		for k, v := range job.Annotations {
			deployment.Annotations[k] = v
		}
		deployment.Annotations["k8s.v6d.io/vineyardjob"] = job.Name
		for k, v := range job.Labels {
			deployment.Labels[k] = v
		}
		if err := ctrl.SetControllerReference(job, deployment, r.Scheme); err != nil {
			r.Log.Error(err, "failed to attach controller reference to deployment")
			return nil, err
		}
		deployment.Spec.Selector = &metav1.LabelSelector{MatchLabels: deployment.Labels}
		deployment.Spec.Template.Labels = deployment.Labels
		return deployment, nil
	}

	deploymentspec, err := makeDeploymentForJob(job)
	if err != nil {
		log.Error(err, "failed to construct deployment for job", "spec", job.Spec.Deployment)
		return ctrl.Result{Requeue: false}, err
	}
	if err := r.Create(ctx, deploymentspec); err != nil {
		log.Error(err, "failed to create deployment for job", "spec", deploymentspec.Spec)
		return ctrl.Result{Requeue: false}, err
	}

	go r.checkDeploymentStatus(ctx, job, target, false)
	return ctrl.Result{}, nil
}

func (r *VineyardJobReconciler) reconcileReplicaSet(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName) (ctrl.Result, error) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	log.Info("start reconcilering replicaset ...")

	var replicaset appsv1.ReplicaSet
	if err := r.Get(ctx, target, &replicaset); err == nil {
		log.V(10).Info("replicaset already exists, it might be a status update request")
		if !r.ifReplicaSetReady(&replicaset) {
			go r.checkReplicaSetStatus(ctx, job, target, true)
		}
		return ctrl.Result{}, nil
	}

	log.Info("start creating replicaset for vineyardjob ...")

	makeReplicaSetForJob := func(job *v1alpha1.VineyardJob) (*appsv1.ReplicaSet, error) {
		replicaset := &appsv1.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				Name:        job.Name,
				Namespace:   job.Namespace,
			},
			Spec: *job.Spec.ReplicaSet.DeepCopy(),
		}
		for k, v := range job.Annotations {
			replicaset.Annotations[k] = v
		}
		replicaset.Annotations["k8s.v6d.io/vineyardjob"] = job.Name
		for k, v := range job.Labels {
			replicaset.Labels[k] = v
		}
		if err := ctrl.SetControllerReference(job, replicaset, r.Scheme); err != nil {
			r.Log.Error(err, "failed to attach controller reference to replicaset")
			return nil, err
		}
		replicaset.Spec.Selector = &metav1.LabelSelector{MatchLabels: replicaset.Labels}
		replicaset.Spec.Template.Labels = replicaset.Labels
		return replicaset, nil
	}

	replicasetspec, err := makeReplicaSetForJob(job)
	if err != nil {
		log.Error(err, "failed to construct replicaset for job", "spec", job.Spec.Deployment)
		return ctrl.Result{Requeue: false}, err
	}
	if err := r.Create(ctx, replicasetspec); err != nil {
		log.Error(err, "failed to create replicaset for job", "spec", replicasetspec.Spec)
		return ctrl.Result{Requeue: false}, err
	}

	go r.checkReplicaSetStatus(ctx, job, target, false)
	return ctrl.Result{}, nil
}

func (r *VineyardJobReconciler) reconcileStatefulSet(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName) (ctrl.Result, error) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	log.Info("start reconcilering statefulset ...")

	var statefulset appsv1.StatefulSet
	if err := r.Get(ctx, target, &statefulset); err == nil {
		log.V(10).Info("statefulset already exists, it might be a status update request")
		if !r.ifStatefulSetReady(&statefulset) {
			go r.checkStatefulSetStatus(ctx, job, target, true)
		}
		return ctrl.Result{}, nil
	}

	log.Info("start creating statefulset for vineyardjob ...")

	makeStatefulSetForJob := func(job *v1alpha1.VineyardJob) (*appsv1.StatefulSet, error) {
		statefulset := &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
				Name:        job.Name,
				Namespace:   job.Namespace,
			},
			Spec: *job.Spec.StatefulSet.DeepCopy(),
		}
		for k, v := range job.Annotations {
			statefulset.Annotations[k] = v
		}
		statefulset.Annotations["k8s.v6d.io/vineyardjob"] = job.Name
		for k, v := range job.Labels {
			statefulset.Labels[k] = v
		}
		if err := ctrl.SetControllerReference(job, statefulset, r.Scheme); err != nil {
			r.Log.Error(err, "failed to attach controller reference to statefulset")
			return nil, err
		}
		statefulset.Spec.Selector = &metav1.LabelSelector{MatchLabels: statefulset.Labels}
		statefulset.Spec.Template.Labels = statefulset.Labels
		return statefulset, nil
	}

	statefulsetspec, err := makeStatefulSetForJob(job)
	if err != nil {
		log.Error(err, "failed to construct statefulset for job", "spec", job.Spec.StatefulSet)
		return ctrl.Result{Requeue: false}, err
	}
	if err := r.Create(ctx, statefulsetspec); err != nil {
		log.Error(err, "failed to create statefulset for job", "spec", statefulsetspec.Spec)
		return ctrl.Result{Requeue: false}, err
	}

	go r.checkStatefulSetStatus(ctx, job, target, false)
	return ctrl.Result{}, nil
}

func (r *VineyardJobReconciler) ifPodReady(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodRunning
}

func (r *VineyardJobReconciler) checkPodStatus(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName, delay bool) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	r.waitForRefresh(delay)
	log.Info("checking pod status ...")

	job.Status.Replica = job.Spec.Replica
	job.Status.Ready = 0
	job.Status.Hosts = make([]string, job.Spec.Replica)

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
	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "failed to update the job status")
	}
}

func (r *VineyardJobReconciler) ifDeploymentReady(deployment *appsv1.Deployment) bool {
	return *deployment.Spec.Replicas == deployment.Status.ReadyReplicas
}

func (r *VineyardJobReconciler) checkDeploymentStatus(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName, delay bool) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	r.waitForRefresh(delay)
	log.Info("checking deployment status ...")

	var deployment appsv1.Deployment

	if err := r.Get(ctx, target, &deployment); err != nil {
		log.Info("failed to get status of deployment", "deployment", target)
		return
	}
	job.Status.Replica = int(*deployment.Spec.Replicas)
	job.Status.Ready = int(deployment.Status.ReadyReplicas)
	if job.Status.Ready == job.Status.Replica {
		job.Status.Hosts = make([]string, job.Status.Ready)
		// for _, loc := range deployment.Status.
		pods := &corev1.PodList{}
		if err := r.List(ctx, pods, client.InNamespace(target.Namespace), client.MatchingLabels(deployment.Spec.Template.Labels)); err != nil {
			log.Info("failed to list pods", "deployment", target)
		}
		for i, pod := range pods.Items {
			job.Status.Hosts[i] = pod.Spec.NodeName
		}
	} else {
		job.Status.Hosts = []string{}
	}
	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "failed to update the job status")
	}
}

func (r *VineyardJobReconciler) ifReplicaSetReady(replicaset *appsv1.ReplicaSet) bool {
	return *replicaset.Spec.Replicas == replicaset.Status.ReadyReplicas
}

func (r *VineyardJobReconciler) checkReplicaSetStatus(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName, delay bool) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	r.waitForRefresh(delay)
	log.Info("checking replicaset status ...")

	var replicaset appsv1.ReplicaSet

	if err := r.Get(ctx, target, &replicaset); err != nil {
		log.Info("failed to get status of replicaset", "replicaset", target)
		return
	}
	job.Status.Replica = int(*replicaset.Spec.Replicas)
	job.Status.Ready = int(replicaset.Status.ReadyReplicas)
	if job.Status.Ready == job.Status.Replica {
		job.Status.Hosts = make([]string, job.Status.Ready)
		// for _, loc := range replicaset.Status.
		pods := &corev1.PodList{}
		if err := r.List(ctx, pods, client.InNamespace(target.Namespace), client.MatchingLabels(replicaset.Spec.Template.Labels)); err != nil {
			log.Info("failed to list pods", "replicaset", target)
		}
		for i, pod := range pods.Items {
			job.Status.Hosts[i] = pod.Spec.NodeName
		}
	} else {
		job.Status.Hosts = []string{}
	}
	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "failed to update the job status")
	}
}

func (r *VineyardJobReconciler) ifStatefulSetReady(statefulset *appsv1.StatefulSet) bool {
	return *statefulset.Spec.Replicas == statefulset.Status.ReadyReplicas
}

func (r *VineyardJobReconciler) checkStatefulSetStatus(ctx context.Context, job *v1alpha1.VineyardJob, target types.NamespacedName, delay bool) {
	log := r.Log.WithValues("vineyardjob", job.Name)

	r.waitForRefresh(delay)
	log.Info("checking statefulset status ...")

	var statefulset appsv1.StatefulSet

	if err := r.Get(ctx, target, &statefulset); err != nil {
		log.Info("failed to get status of statefulset", "statefulset", target)
		return
	}
	job.Status.Replica = int(*statefulset.Spec.Replicas)
	job.Status.Ready = int(statefulset.Status.ReadyReplicas)
	if job.Status.Ready == job.Status.Replica {
		job.Status.Hosts = make([]string, job.Status.Ready)
		// for _, loc := range statefulset.Status.
		pods := &corev1.PodList{}
		if err := r.List(ctx, pods, client.InNamespace(target.Namespace), client.MatchingLabels(statefulset.Spec.Template.Labels)); err != nil {
			log.Info("failed to list pods", "statefulset", target)
		}
		for i, pod := range pods.Items {
			job.Status.Hosts[i] = pod.Spec.NodeName
		}
	} else {
		job.Status.Hosts = []string{}
	}
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
