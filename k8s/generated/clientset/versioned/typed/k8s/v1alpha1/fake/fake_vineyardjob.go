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
// Code generated by client-gen. DO NOT EDIT.

package fake

import (
	"context"

	v1alpha1 "github.com/v6d-io/v6d/k8s/api/k8s/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeVineyardJobs implements VineyardJobInterface
type FakeVineyardJobs struct {
	Fake *FakeK8sV1alpha1
	ns   string
}

var vineyardjobsResource = schema.GroupVersionResource{Group: "k8s", Version: "v1alpha1", Resource: "vineyardjobs"}

var vineyardjobsKind = schema.GroupVersionKind{Group: "k8s", Version: "v1alpha1", Kind: "VineyardJob"}

// Get takes name of the vineyardJob, and returns the corresponding vineyardJob object, and an error if there is any.
func (c *FakeVineyardJobs) Get(ctx context.Context, name string, options v1.GetOptions) (result *v1alpha1.VineyardJob, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(vineyardjobsResource, c.ns, name), &v1alpha1.VineyardJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.VineyardJob), err
}

// List takes label and field selectors, and returns the list of VineyardJobs that match those selectors.
func (c *FakeVineyardJobs) List(ctx context.Context, opts v1.ListOptions) (result *v1alpha1.VineyardJobList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(vineyardjobsResource, vineyardjobsKind, c.ns, opts), &v1alpha1.VineyardJobList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.VineyardJobList{ListMeta: obj.(*v1alpha1.VineyardJobList).ListMeta}
	for _, item := range obj.(*v1alpha1.VineyardJobList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested vineyardJobs.
func (c *FakeVineyardJobs) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(vineyardjobsResource, c.ns, opts))

}

// Create takes the representation of a vineyardJob and creates it.  Returns the server's representation of the vineyardJob, and an error, if there is any.
func (c *FakeVineyardJobs) Create(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.CreateOptions) (result *v1alpha1.VineyardJob, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(vineyardjobsResource, c.ns, vineyardJob), &v1alpha1.VineyardJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.VineyardJob), err
}

// Update takes the representation of a vineyardJob and updates it. Returns the server's representation of the vineyardJob, and an error, if there is any.
func (c *FakeVineyardJobs) Update(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.UpdateOptions) (result *v1alpha1.VineyardJob, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(vineyardjobsResource, c.ns, vineyardJob), &v1alpha1.VineyardJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.VineyardJob), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeVineyardJobs) UpdateStatus(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.UpdateOptions) (*v1alpha1.VineyardJob, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(vineyardjobsResource, "status", c.ns, vineyardJob), &v1alpha1.VineyardJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.VineyardJob), err
}

// Delete takes name of the vineyardJob and deletes it. Returns an error if one occurs.
func (c *FakeVineyardJobs) Delete(ctx context.Context, name string, opts v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(vineyardjobsResource, c.ns, name), &v1alpha1.VineyardJob{})

	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeVineyardJobs) DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(vineyardjobsResource, c.ns, listOpts)

	_, err := c.Fake.Invokes(action, &v1alpha1.VineyardJobList{})
	return err
}

// Patch applies the patch and returns the patched vineyardJob.
func (c *FakeVineyardJobs) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.VineyardJob, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(vineyardjobsResource, c.ns, name, pt, data, subresources...), &v1alpha1.VineyardJob{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.VineyardJob), err
}
