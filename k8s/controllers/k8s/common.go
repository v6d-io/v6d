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

// Package k8s provides common functions for k8s controllers.
package k8s

import (
	"context"

	"github.com/pkg/errors"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ApplyStatueUpdate applies the status update to the object.
func ApplyStatueUpdate[Obj client.Object](ctx context.Context, c client.Client,
	o Obj, w client.StatusWriter, overlayObj func(Obj) (error, Obj)) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		name := client.ObjectKeyFromObject(o)
		if err := c.Get(ctx, name, o); err != nil {
			return errors.Wrap(err, "failed to get "+name.String())
		}
		// Overlay the object with the new status.
		err, newObj := overlayObj(o)
		if err != nil {
			return errors.Wrap(err, "failed to overlay "+name.String())
		}

		if err := w.Update(ctx, newObj); err != nil {
			return errors.Wrap(err, "failed to update sidecar's status")
		}
		return nil
	})
}
