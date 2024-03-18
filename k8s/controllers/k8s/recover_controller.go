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
	"bytes"
	"context"
	"io"
	"strings"
	"time"

	"github.com/pkg/errors"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	swckkube "github.com/apache/skywalking-swck/operator/pkg/kubernetes"

	k8sv1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	v1alpha1 "github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	"github.com/v6d-io/v6d/k8s/pkg/log"
	"github.com/v6d-io/v6d/k8s/pkg/templates"
	"k8s.io/client-go/kubernetes"
)

const (
	// SucceedState is the succeed state
	SucceedState = "Succeed"
	// RunningState is the running state
	RunningState = "Running"
)

// RecoverReconciler reconciles a Recover object
type RecoverReconciler struct {
	client.Client
	*kubernetes.Clientset
	Scheme *runtime.Scheme
}

// RecoverConfig holds all configuration about recover
type RecoverConfig struct {
	Name string
	FailoverConfig
}

// +kubebuilder:rbac:groups=k8s.v6d.io,resources=recovers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=recovers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=vineyardds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;create;update;delete
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get;create;update;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=backups,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=k8s.v6d.io,resources=backups/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;create;update;delete
// +kubebuilder:rbac:groups="",resources=pods/status,verbs=get;list;create;update;delete
// +kubebuilder:rbac:groups="",resources=pods/log,verbs=get

// Reconcile reconciles the Recover.
func (r *RecoverReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithName("controllers").WithName("Recover")

	recover := k8sv1alpha1.Recover{}
	if err := r.Get(ctx, req.NamespacedName, &recover); err != nil {
		logger.Error(err, "unable to fetch Recover")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	logger.Info("Reconciling Recover", "recover", recover)

	backup := k8sv1alpha1.Backup{}
	if err := r.Get(ctx, client.ObjectKey{
		Namespace: recover.Spec.BackupNamespace,
		Name:      recover.Spec.BackupName,
	}, &backup); err != nil {
		logger.Error(err, "unable to get Backup")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// if pvcName is empty, we'll create a pv and pvc here.
	pvcName := ""
	opts := NewRecoverOpts(
		"recover-"+backup.Name,
		pvcName,
		backup.Spec.BackupPath,
	)
	recoverCfg, err := opts.BuildCfgForController(r.Client, &backup)
	if err != nil {
		logger.Error(err, "failed to build recover config")
		return ctrl.Result{}, err
	}

	app := swckkube.Application{
		Client:   r.Client,
		FileRepo: templates.Repo,
		CR:       &recover,
		GVK:      k8sv1alpha1.GroupVersion.WithKind("Recover"),
		TmplFunc: map[string]interface{}{"getRecoverConfig": func() RecoverConfig {
			return recoverCfg
		}},
	}

	if recover.Status.State == "" || recover.Status.State == RunningState {
		if _, err := app.Apply(ctx, "recover/job.yaml", logger, false); err != nil {
			logger.Error(err, "failed to apply recover job")
			return ctrl.Result{}, err
		}
		if err := r.UpdateStateStatus(ctx, &backup, &recover); err != nil {
			logger.Error(err, "failed to update recover state status")
			return ctrl.Result{}, err
		}
	}

	if recover.Status.State == SucceedState {
		if err := r.UpdateMappingStatus(ctx, &backup, &recover); err != nil {
			logger.Error(err, "failed to update recover objectmapping status")
			return ctrl.Result{}, err
		}
	}

	// reconcile every minute
	return ctrl.Result{RequeueAfter: time.Minute}, nil
}

// UpdateStateStatus updates the state status of the Recover.
func (r *RecoverReconciler) UpdateStateStatus(
	ctx context.Context,
	backup *k8sv1alpha1.Backup,
	recover *k8sv1alpha1.Recover,
) error {
	name := client.ObjectKey{Name: "recover-" + backup.Name, Namespace: backup.Namespace}
	job := batchv1.Job{}
	if err := r.Get(ctx, name, &job); err != nil {
		log.V(1).Error(err, "failed to get job")
	}

	// get job state
	state := RunningState
	if job.Status.Succeeded == *job.Spec.Parallelism {
		state = SucceedState
	}

	status := &k8sv1alpha1.RecoverStatus{
		State: state,
	}
	if err := r.applyStatusUpdate(ctx, recover, status); err != nil {
		return errors.Wrap(err, "failed to update status")
	}
	return nil
}

// UpdateMappingStatus updates the mapping status of the Recover.
func (r *RecoverReconciler) UpdateMappingStatus(
	ctx context.Context,
	backup *k8sv1alpha1.Backup,
	recover *k8sv1alpha1.Recover,
) error {
	jobName := "recover-" + backup.Name
	jobNamespace := backup.Namespace
	objectMapping, err := GetObjectMappingTable(r.Client, *r.Clientset, jobName, jobNamespace)
	if err != nil {
		return errors.Wrap(err, "failed to get object mapping table")
	}

	status := &k8sv1alpha1.RecoverStatus{
		ObjectMapping: objectMapping,
		State:         recover.Status.State,
	}

	if err := r.applyStatusUpdate(ctx, recover, status); err != nil {
		return errors.Wrap(err, "failed to update status")
	}
	return nil
}

func (r *RecoverReconciler) applyStatusUpdate(ctx context.Context,
	recover *k8sv1alpha1.Recover, status *k8sv1alpha1.RecoverStatus,
) error {
	return ApplyStatueUpdate(ctx, r.Client, recover, r.Status(),
		func(recover *v1alpha1.Recover) (error, *v1alpha1.Recover) {
			recover.Status = *status
			recover.Kind = "Recover"
			if err := swckkube.ApplyOverlay(recover, &k8sv1alpha1.Recover{Status: *status}); err != nil {
				return errors.Wrap(err, "failed to overlay recover's status"), nil
			}
			return nil, recover
		},
	)
}

// SetupWithManager sets up the controller with the Manager.
func (r *RecoverReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&k8sv1alpha1.Recover{}).
		Complete(r)
}

// RecoverOpts holds all options of recover config
type RecoverOpts struct {
	name    string
	pvcName string
	path    string
}

func NewRecoverOpts(name, pvcName, path string) RecoverOpts {
	return RecoverOpts{
		name:    name,
		pvcName: pvcName,
		path:    path,
	}
}

func (opts RecoverOpts) BuildCfgForVineyarctl(c client.Client, backup *k8sv1alpha1.Backup) (RecoverConfig, error) {
	withScheduler := false
	return buildRecoverCfg(c, backup, opts, withScheduler)
}

func (opts RecoverOpts) BuildCfgForController(c client.Client, backup *k8sv1alpha1.Backup) (RecoverConfig, error) {
	withScheduler := true
	return buildRecoverCfg(c, backup, opts, withScheduler)
}

func buildRecoverCfg(c client.Client, backup *k8sv1alpha1.Backup,
	opts RecoverOpts, withScheduler bool) (RecoverConfig, error) {
	// setup the recover configuration
	recoverCfg := RecoverConfig{}
	recoverCfg.Name = opts.name

	pN := backup.Name
	if opts.pvcName != "" {
		pN = opts.pvcName
	}
	failoverCfg, err := newFailoverConfig(c, backup, opts.path, withScheduler, pN)
	if err != nil {
		return recoverCfg, errors.Wrap(err, "failed to build failover configuration")
	}
	recoverCfg.FailoverConfig = failoverCfg

	return recoverCfg, nil
}

// getObjectMappingFromPodLogs gets object mapping from pod logs.
func getObjectMappingFromPodLogs(
	clientset *kubernetes.Clientset,
	pod *corev1.Pod,
) (map[string]string, error) {
	mappingtable := make(map[string]string)

	req := clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{})
	logs, err := req.Stream(context.Background())
	if err != nil {
		return mappingtable, errors.Wrap(err, "failed to open stream")
	}
	defer logs.Close()

	buf := new(bytes.Buffer)
	_, err = io.Copy(buf, logs)
	if err != nil {
		return mappingtable, errors.Wrap(err, "failed to copy logs")
	}
	log := buf.String()

	objmappings := strings.Split(log, "\n")
	for _, s := range objmappings {
		mapping := strings.TrimSpace(s)
		if strings.Contains(mapping, "->") {
			result := strings.Split(mapping, "->")
			mappingtable[result[0]] = result[1]
		}
	}
	return mappingtable, nil
}

func GetObjectMappingTable(c client.Client, clientset kubernetes.Clientset,
	jobName, jobNamespace string) (map[string]string, error) {
	objectMapping := make(map[string]string)
	name := client.ObjectKey{Name: jobName, Namespace: jobNamespace}
	job := batchv1.Job{}
	err := c.Get(context.Background(), name, &job)
	if err != nil {
		return objectMapping, errors.Wrap(err, "failed to get job")
	}
	// if job completd and deleted after ttl
	if apierrors.IsNotFound(err) {
		return objectMapping, nil
	}

	// get object mappings
	labels := job.Spec.Template.Labels
	podList := &corev1.PodList{}
	opts := []client.ListOption{
		client.MatchingLabels{
			"controller-uid": labels["controller-uid"],
		},
	}
	if err := c.List(context.Background(), podList, opts...); err != nil {
		return objectMapping, errors.Wrap(err, "failed to list pod created by job")
	}

	for i := range podList.Items {
		mapping, err := getObjectMappingFromPodLogs(&clientset, &podList.Items[i])
		if err != nil {
			return objectMapping, errors.Wrap(err, "failed to get logs from pods")
		}
		for k, v := range mapping {
			objectMapping[k] = v
		}
	}
	return objectMapping, nil
}
