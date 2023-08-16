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
var olog = log.WithName("webhook").WithName("operation")

// SetupWebhookWithManager implements the webhook.Defaulter so a webhook will be registered
func (r *Operation) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

//nolint: lll
//+kubebuilder:webhook:path=/mutate-k8s-v6d-io-v1alpha1-operation,mutating=true,failurePolicy=fail,sideEffects=None,groups=k8s.v6d.io,resources=operations,verbs=create;update,versions=v1alpha1,name=moperation.kb.io,admissionReviewVersions=v1

var _ webhook.Defaulter = &Operation{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (r *Operation) Default() {
	olog.Info("default", "name", r.Name)
	// set the default timeoutSeconds
	if r.Spec.TimeoutSeconds == 0 {
		r.Spec.TimeoutSeconds = 300
	}
	r.check()
}

//nolint: lll
//+kubebuilder:webhook:path=/validate-k8s-v6d-io-v1alpha1-operation,mutating=false,failurePolicy=fail,sideEffects=None,groups=k8s.v6d.io,resources=operations,verbs=create;update,versions=v1alpha1,name=voperation.kb.io,admissionReviewVersions=v1

var _ webhook.Validator = &Operation{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *Operation) ValidateCreate() error {
	olog.Info("validate create", "name", r.Name)
	r.check()

	return nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *Operation) ValidateUpdate(old runtime.Object) error {
	olog.Info("validate update", "name", r.Name)
	r.check()

	return nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *Operation) ValidateDelete() error {
	olog.Info("validate delete", "name", r.Name)

	return nil
}

func (r *Operation) check() {
	if r.Spec.Name == "" {
		olog.Error(nil, "operation's name is absent")
	} else if r.Spec.Name != "assembly" && r.Spec.Name != "repartition" {
		olog.Error(nil, "operation's name is invalid")
	}
	if r.Spec.Type == "" {
		olog.Error(nil, "operation's type is absent")
	}
	if r.Spec.Require == "" {
		olog.Error(nil, "operation's require is absent")
	}
	if r.Spec.Target == "" {
		olog.Error(nil, "operation's target is absent")
	}
}
