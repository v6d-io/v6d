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

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

// log is for logging in this package.
var localobjectlog = logf.Log.WithName("localobject-resource")

func (r *LocalObject) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!

//+kubebuilder:webhook:path=/mutate-k8s-v6d-io-v1alpha1-localobject,mutating=true,failurePolicy=fail,groups=k8s.v6d.io,resources=localobjects,verbs=create;update,versions=v1alpha1,admissionReviewVersions=v1;v1beta1,sideEffects=NoneOnDryRun,name=mlocalobject.kb.io

var _ webhook.Defaulter = &LocalObject{}

// Default implements webhook.Defaulter so a webhook will be registered for the type
func (r *LocalObject) Default() {
	localobjectlog.Info("default", "name", r.Name)

	// TODO(user): fill in your defaulting logic.
}

// TODO(user): change verbs to "verbs=create;update;delete" if you want to enable deletion validation.
//+kubebuilder:webhook:verbs=create;update,path=/validate-k8s-v6d-io-v1alpha1-localobject,mutating=false,failurePolicy=fail,groups=k8s.v6d.io,resources=localobjects,versions=v1alpha1,admissionReviewVersions=v1;v1beta1,sideEffects=NoneOnDryRun,name=vlocalobject.kb.io

var _ webhook.Validator = &LocalObject{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *LocalObject) ValidateCreate() error {
	localobjectlog.Info("validate create", "name", r.Name)

	// TODO(user): fill in your validation logic upon object creation.
	return nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *LocalObject) ValidateUpdate(old runtime.Object) error {
	localobjectlog.Info("validate update", "name", r.Name)

	// TODO(user): fill in your validation logic upon object update.
	return nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *LocalObject) ValidateDelete() error {
	localobjectlog.Info("validate delete", "name", r.Name)

	// TODO(user): fill in your validation logic upon object deletion.
	return nil
}
