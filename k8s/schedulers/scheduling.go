/*
Copyright 2020 The Kubernetes Authors.
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

package schedulers

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubernetes "k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/go-logr/logr"
	v1alpha1 "github.com/v6d-io/v6d/k8s/api/k8s/v1alpha1"
	clientset "github.com/v6d-io/v6d/k8s/generated/clientset/versioned"
	clientsetv1alpha1 "github.com/v6d-io/v6d/k8s/generated/clientset/versioned/typed/k8s/v1alpha1"

	_ "github.com/v6d-io/v6d/k8s/generated/informers/externalversions/k8s/v1alpha1"
	_ "github.com/v6d-io/v6d/k8s/generated/listers/k8s/v1alpha1"
)

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = "Vineyard"
	// Timeout is the default timeout for the scheduler plugin.
	Timeout = 60
	// VineyardJobName is the pod group name
	VineyardJobName = "scheduling.k8s.v6d.io/job"
	// VineyardJobRequired is the object ids that required by this job
	VineyardJobRequired = "scheduling.k8s.v6d.io/required"
	// VineyardJobReplica is the replication of pods in this job.
	VineyardJobReplica = "scheduling.k8s.v6d.io/replica"
)

type VineyardScheduler struct {
	Log           logr.Logger
	Kubernetes    *kubernetes.Clientset
	Clientset     *clientset.Clientset
	LocalClients  map[string]clientsetv1alpha1.LocalObjectInterface
	GlobalClients map[string]clientsetv1alpha1.GlobalObjectInterface
}

func New(log logr.Logger) *VineyardScheduler {
	Clientset := clientset.NewForConfigOrDie(ctrl.GetConfigOrDie())
	Kubernetes := kubernetes.NewForConfigOrDie(ctrl.GetConfigOrDie())

	scheduler := &VineyardScheduler{
		Log:           log.WithName("scheduler"),
		Clientset:     Clientset,
		Kubernetes:    Kubernetes,
		LocalClients:  make(map[string]clientsetv1alpha1.LocalObjectInterface),
		GlobalClients: make(map[string]clientsetv1alpha1.GlobalObjectInterface),
	}
	return scheduler
}

func (sched *VineyardScheduler) GetAllNodes(ctx context.Context) ([]string, error) {
	if nodes, err := sched.Kubernetes.CoreV1().Nodes().List(ctx, metav1.ListOptions{}); err != nil {
		return nil, err
	} else {
		nodenames := make([]string, nodes.Size())
		for index, node := range nodes.Items {
			nodenames[index] = node.Name
		}
		return nodenames, nil
	}
}

func (sched *VineyardScheduler) getGlobalObjectsByID(ctx context.Context, namespace string, objectIds []string) ([]v1alpha1.GlobalObject, error) {
	objects := make([]v1alpha1.GlobalObject, 0)
	client := sched.GlobalClients[namespace]
	for _, globalObjectID := range objectIds {
		if globalObject, err := client.Get(ctx, globalObjectID, metav1.GetOptions{}); err != nil {
			return nil, err
		} else {
			objects = append(objects, *globalObject)
		}
	}
	return objects, nil
}

func (sched *VineyardScheduler) getLocalObjectsBySignatures(ctx context.Context, namespace string, signatures []string) ([]v1alpha1.LocalObject, error) {
	objects := make([]v1alpha1.LocalObject, 0)
	client := sched.LocalClients[namespace]
	for _, sig := range signatures {
		options := metav1.ListOptions{
			LabelSelector: fmt.Sprintf("k8s.v6d.io/signature=%v", sig),
		}
		if localObjects, err := client.List(ctx, options); err != nil {
			return nil, err
		} else {
			objects = append(objects, localObjects.Items...)
		}
	}
	return objects, nil
}

func (sched *VineyardScheduler) setupForNamespace(namespace string) {
	if _, ok := sched.GlobalClients[namespace]; ok {
		return
	}
	sched.GlobalClients[namespace] = sched.Clientset.K8sV1alpha1().GlobalObjects(namespace)

	if _, ok := sched.LocalClients[namespace]; ok {
		return
	}
	sched.LocalClients[namespace] = sched.Clientset.K8sV1alpha1().LocalObjects(namespace)
}

func (sched *VineyardScheduler) ComputePlacement(ctx context.Context, namespace string, requires []string, replica int) ([]string, [][]string, error) {
	if len(requires) == 0 {
		return nil, nil, fmt.Errorf("No 'requires' arguments")
	}

	sched.setupForNamespace(namespace)

	// 1. find global objects
	globalObjects, err := sched.getGlobalObjectsByID(ctx, namespace, requires)
	if err != nil {
		return nil, nil, err
	}
	sched.Log.Info("required", "global", globalObjects)

	// 2. find local objects
	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := sched.getLocalObjectsBySignatures(ctx, namespace, localsigs)
	if err != nil {
		return nil, nil, err
	}
	if len(localObjects) == 0 {
		return nil, nil, fmt.Errorf("No local chunks found")
	}
	sched.Log.Info("required", "local chunks", localObjects)

	// 3. group local objects based on hosts
	locations := make(map[string][]string)
	for _, localObject := range localObjects {
		host := localObject.Spec.Hostname
		if _, ok := locations[host]; !ok {
			locations[host] = make([]string, 0)
		}
		locations[host] = append(locations[host], localObject.Spec.ObjectID)
	}

	// total frags
	totalfrags := len(localObjects)
	// frags for per pod
	chunks_per_worker := totalfrags / replica
	if totalfrags%replica != 0 {
		chunks_per_worker++
	}

	// 4. sort all candidate nodes
	hosts := make([]string, 0)
	for k := range locations {
		hosts = append(hosts, k)
	}
	sort.Strings(hosts)

	// 5. sort all local chunks for each node, and concatente
	sortedfrags := make([]string, 0)
	fraglocations := make(map[string]string)
	for _, host := range hosts {
		frags := locations[host]
		sort.Strings(frags)

		sortedfrags = append(sortedfrags, frags...)
		for _, frag := range frags {
			fraglocations[frag] = host
		}
	}

	placement := make([]string, 0)
	fragsplits := make([][]string, 0)

	// 6. generate chunks that assigned for each worker
	frags := make([]string, 0)
	for _, frag := range sortedfrags {
		if len(frags) == chunks_per_worker {
			fragsplits = append(fragsplits, frags)
			frags = make([]string, 0)
		} else {
			frags = append(frags, frag)
		}
	}

	// 7. generate chunks placement for each chunk splits
	for _, frags := range fragsplits {
		stats := make(map[string]int)
		for _, frag := range frags {
			loc := fraglocations[frag]
			if cnt, ok := stats[frag]; ok {
				stats[loc] = cnt + 1
			} else {
				stats[loc] = 1
			}
		}

		var maxcnt int = 0
		var target string = "undefined" // indicates an error
		for host, cnt := range stats {
			if cnt > maxcnt {
				maxcnt = cnt
				target = host
			}
		}
		placement = append(placement, target)
	}

	return placement, fragsplits, nil
}

func (sched *VineyardScheduler) ComputePlacementFor(ctx context.Context, job *v1alpha1.VineyardJob) ([]string, [][]string, error) {
	var requires []string
	var replica int
	if value, ok := job.Labels[VineyardJobRequired]; ok {
		requires = strings.Split(value, "-")
	} else {
		return nil, nil, fmt.Errorf("Failed to parse 'required' for job")
	}
	if value, err := sched.ParseReplicas(job); err != nil {
		return nil, nil, err
	} else {
		replica = value
	}
	return sched.ComputePlacement(ctx, job.Namespace, requires, replica)
}

func (sched *VineyardScheduler) ParseReplicas(job *v1alpha1.VineyardJob) (int, error) {
	if value, ok := job.Labels[VineyardJobReplica]; ok {
		if intvalue, err := strconv.Atoi(value); err != nil {
			return -1, err
		} else {
			return intvalue, nil
		}
	}
	return job.Spec.Replicas, nil
}
