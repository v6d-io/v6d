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

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	"github.com/v6d-io/v6d/k8s/pkg/log"
)

// log is for logging in this package.
var llog = log.WithName("webhook").WithName("localobject")

// SetupWebhookWithManager implements the webhook.Defaulter so a webhook will be registered
func (r *LocalObject) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

//nolint: lll
//+kubebuilder:webhook:path=/mutate-k8s-v6d-io-v1alpha1-localobject,mutating=true,failurePolicy=fail,groups=k8s.v6d.io,resources=localobjects,verbs=create;update,versions=v1alpha1,admissionReviewVersions=v1,sideEffects=None,name=mlocalobject.kb.io

var _ webhook.Defaulter = &LocalObject{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (r *LocalObject) Default() {
	llog.Info("default", "name", r.Name)
}

//nolint: lll
//+kubebuilder:webhook:verbs=create;update,path=/validate-k8s-v6d-io-v1alpha1-localobject,mutating=false,failurePolicy=fail,groups=k8s.v6d.io,resources=localobjects,versions=v1alpha1,admissionReviewVersions=v1,sideEffects=None,name=vlocalobject.kb.io

var _ webhook.Validator = &LocalObject{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *LocalObject) ValidateCreate() error {
	llog.Info("validate create", "name", r.Name)

	return nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *LocalObject) ValidateUpdate(old runtime.Object) error {
	llog.Info("validate update", "name", r.Name)

	return nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *LocalObject) ValidateDelete() error {
	llog.Info("validate delete", "name", r.Name)

	return nil
}
