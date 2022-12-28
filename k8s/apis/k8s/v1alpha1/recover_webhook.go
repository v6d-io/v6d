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
var recoverlog = logf.Log.WithName("recover-resource")

// SetupWebhookWithManager implements the webhook.Defaulter so a webhook will be registered
func (r *Recover) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

// TODO(user): EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!

//+kubebuilder:webhook:path=/mutate-k8s-v6d-io-v1alpha1-recover,mutating=true,failurePolicy=fail,sideEffects=None,groups=k8s.v6d.io,resources=recovers,verbs=create;update,versions=v1alpha1,name=mrecover.kb.io,admissionReviewVersions=v1

var _ webhook.Defaulter = &Recover{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (r *Recover) Default() {
	recoverlog.Info("default", "name", r.Name)

	// TODO(user): fill in your defaulting logic.
}

// TODO(user): change verbs to "verbs=create;update;delete" if you want to enable deletion validation.
//+kubebuilder:webhook:path=/validate-k8s-v6d-io-v1alpha1-recover,mutating=false,failurePolicy=fail,sideEffects=None,groups=k8s.v6d.io,resources=recovers,verbs=create;update,versions=v1alpha1,name=vrecover.kb.io,admissionReviewVersions=v1

var _ webhook.Validator = &Recover{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *Recover) ValidateCreate() error {
	recoverlog.Info("validate create", "name", r.Name)

	// TODO(user): fill in your validation logic upon object creation.
	return nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *Recover) ValidateUpdate(old runtime.Object) error {
	recoverlog.Info("validate update", "name", r.Name)

	// TODO(user): fill in your validation logic upon object update.
	return nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *Recover) ValidateDelete() error {
	recoverlog.Info("validate delete", "name", r.Name)

	// TODO(user): fill in your validation logic upon object deletion.
	return nil
}
