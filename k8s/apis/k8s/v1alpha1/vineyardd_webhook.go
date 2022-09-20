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

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

// log is for logging in this package.
var vineyarddlog = logf.Log.WithName("vineyardd-resource")

// SetupWebhookWithManager implements the webhook.Defaulter so a webhook will be registered
func (r *Vineyardd) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

//nolint: lll
//+kubebuilder:webhook:path=/mutate-k8s-v6d-io-v1alpha1-vineyardd,mutating=true,failurePolicy=fail,sideEffects=None,groups=k8s.v6d.io,resources=vineyardds,verbs=create;update,versions=v1alpha1,name=mvineyardd.kb.io,admissionReviewVersions=v1

var _ webhook.Defaulter = &Vineyardd{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (r *Vineyardd) Default() {
	vineyarddlog.Info("default", "name", r.Name)
	// check the image、version and replicas
	r.check()
}

//nolint: lll
//+kubebuilder:webhook:path=/validate-k8s-v6d-io-v1alpha1-vineyardd,mutating=false,failurePolicy=fail,sideEffects=None,groups=k8s.v6d.io,resources=vineyardds,verbs=create;update,versions=v1alpha1,name=vvineyardd.kb.io,admissionReviewVersions=v1

var _ webhook.Validator = &Vineyardd{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *Vineyardd) ValidateCreate() error {
	vineyarddlog.Info("validate create", "name", r.Name)
	// check the image、version and replicas
	r.check()

	return nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *Vineyardd) ValidateUpdate(old runtime.Object) error {
	vineyarddlog.Info("validate update", "name", r.Name)
	// check the image、version and replicas
	r.check()

	return nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *Vineyardd) ValidateDelete() error {
	vineyarddlog.Info("validate delete", "name", r.Name)

	return nil
}

// Make sure the vineyardd is valid
func (r *Vineyardd) check() {
	vineyarddlog.Info("validate image", "name", r.Name)
	defaultImage := "docker.pkg.github.com/v6d-io/v6d/vineyardd:alpine-latest"
	defaultVersion := "0.9.0"
	defaultReplicas := 3
	if r.Spec.Image == "" {
		vineyarddlog.Info("the image is absent, use the default image", "name", defaultImage)
		r.Spec.Image = defaultImage
	}
	if r.Spec.Version == "" {
		vineyarddlog.Info("the version is absent, use the default version", "name", defaultVersion)
		r.Spec.Version = defaultVersion
	}
	if r.Spec.Replicas == 0 {
		vineyarddlog.Info("the replicas is absent, use the default replicas", "name", 1)
		r.Spec.Replicas = defaultReplicas
	}
}
