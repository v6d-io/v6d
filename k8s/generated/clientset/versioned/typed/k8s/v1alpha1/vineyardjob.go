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

package v1alpha1

import (
	"context"
	"time"

	v1alpha1 "github.com/v6d-io/v6d/k8s/api/k8s/v1alpha1"
	scheme "github.com/v6d-io/v6d/k8s/generated/clientset/versioned/scheme"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// VineyardJobsGetter has a method to return a VineyardJobInterface.
// A group's client should implement this interface.
type VineyardJobsGetter interface {
	VineyardJobs(namespace string) VineyardJobInterface
}

// VineyardJobInterface has methods to work with VineyardJob resources.
type VineyardJobInterface interface {
	Create(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.CreateOptions) (*v1alpha1.VineyardJob, error)
	Update(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.UpdateOptions) (*v1alpha1.VineyardJob, error)
	UpdateStatus(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.UpdateOptions) (*v1alpha1.VineyardJob, error)
	Delete(ctx context.Context, name string, opts v1.DeleteOptions) error
	DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error
	Get(ctx context.Context, name string, opts v1.GetOptions) (*v1alpha1.VineyardJob, error)
	List(ctx context.Context, opts v1.ListOptions) (*v1alpha1.VineyardJobList, error)
	Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error)
	Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.VineyardJob, err error)
	VineyardJobExpansion
}

// vineyardJobs implements VineyardJobInterface
type vineyardJobs struct {
	client rest.Interface
	ns     string
}

// newVineyardJobs returns a VineyardJobs
func newVineyardJobs(c *K8sV1alpha1Client, namespace string) *vineyardJobs {
	return &vineyardJobs{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the vineyardJob, and returns the corresponding vineyardJob object, and an error if there is any.
func (c *vineyardJobs) Get(ctx context.Context, name string, options v1.GetOptions) (result *v1alpha1.VineyardJob, err error) {
	result = &v1alpha1.VineyardJob{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("vineyardjobs").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do(ctx).
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of VineyardJobs that match those selectors.
func (c *vineyardJobs) List(ctx context.Context, opts v1.ListOptions) (result *v1alpha1.VineyardJobList, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &v1alpha1.VineyardJobList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("vineyardjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested vineyardJobs.
func (c *vineyardJobs) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("vineyardjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Watch(ctx)
}

// Create takes the representation of a vineyardJob and creates it.  Returns the server's representation of the vineyardJob, and an error, if there is any.
func (c *vineyardJobs) Create(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.CreateOptions) (result *v1alpha1.VineyardJob, err error) {
	result = &v1alpha1.VineyardJob{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("vineyardjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(vineyardJob).
		Do(ctx).
		Into(result)
	return
}

// Update takes the representation of a vineyardJob and updates it. Returns the server's representation of the vineyardJob, and an error, if there is any.
func (c *vineyardJobs) Update(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.UpdateOptions) (result *v1alpha1.VineyardJob, err error) {
	result = &v1alpha1.VineyardJob{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("vineyardjobs").
		Name(vineyardJob.Name).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(vineyardJob).
		Do(ctx).
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *vineyardJobs) UpdateStatus(ctx context.Context, vineyardJob *v1alpha1.VineyardJob, opts v1.UpdateOptions) (result *v1alpha1.VineyardJob, err error) {
	result = &v1alpha1.VineyardJob{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("vineyardjobs").
		Name(vineyardJob.Name).
		SubResource("status").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(vineyardJob).
		Do(ctx).
		Into(result)
	return
}

// Delete takes name of the vineyardJob and deletes it. Returns an error if one occurs.
func (c *vineyardJobs) Delete(ctx context.Context, name string, opts v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("vineyardjobs").
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *vineyardJobs) DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("vineyardjobs").
		VersionedParams(&listOpts, scheme.ParameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched vineyardJob.
func (c *vineyardJobs) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.VineyardJob, err error) {
	result = &v1alpha1.VineyardJob{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("vineyardjobs").
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
