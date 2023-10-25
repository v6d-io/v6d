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

// Package schedulers implements the vineyard scheduler plugin.
package schedulers

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"go.uber.org/multierr"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/apis/k8s/v1alpha1"
	ctlclient "github.com/v6d-io/v6d/k8s/cmd/commands/client"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

// SchedulerStrategy is the interface for all scheduler strategies.
type SchedulerStrategy interface {
	// Compute returns the score of each node.
	Compute(interface{}) (interface{}, error)
}

// RoundRobinStrategy is the round robin strategy.
type RoundRobinStrategy struct {
	nodes []string
}

// NewRoundRobinStrategy returns a new RoundRobinStrategy.
func NewRoundRobinStrategy(nodes []string) *RoundRobinStrategy {
	return &RoundRobinStrategy{
		nodes: nodes,
	}
}

// Compute returns the node by the given rank.
func (r *RoundRobinStrategy) Compute(rank int) (string, error) {
	l := len(r.nodes)
	if l == 0 {
		return "", errors.Errorf("No nodes found")
	}
	return r.nodes[rank%l], nil
}

// BestEffortStrategy is the best effort strategy.
type BestEffortStrategy struct {
	client.Client
	// the required jobs
	required []string
	// the replicas of the job(pod or workload)
	replica int
	// the namespace of the job(pod or workload)
	namespace string
	// the ownerReference of created configmap
	ownerReference *[]metav1.OwnerReference
	// hostname -> localobject id
	locations map[string][]string
	// jobname -> globalobject id
	jobGlobalObjectIDs map[string][]string
	nchunks            int
	nodes              []string
}

// NewBestEffortStrategy returns a new BestEffortStrategy.
func NewBestEffortStrategy(
	client client.Client,
	required []string,
	replica int,
	namespace string,
	ownerReference *[]metav1.OwnerReference,
) *BestEffortStrategy {
	return &BestEffortStrategy{
		Client:         client,
		required:       required,
		replica:        replica,
		namespace:      namespace,
		ownerReference: ownerReference,
	}
}

func (b *BestEffortStrategy) TrackingChunksByCRD() *BestEffortStrategy {
	var errList error

	// accumulates all global required objects
	globalObjects, err := b.GetGlobalObjectsByID(b.required)
	if err != nil {
		_ = multierr.Append(errList, err)
	}

	localsigs := make([]string, 0)
	for _, globalObject := range globalObjects {
		localsigs = append(localsigs, globalObject.Spec.Members...)
	}
	localObjects, err := b.GetLocalObjectsBySignatures(localsigs)
	if err != nil {
		_ = multierr.Append(errList, err)
	}

	if len(localObjects) == 0 {
		if len(globalObjects) == 0 {
			localObjects, err = b.GetLocalObjectsByID(b.required)
			// if there is no local objects, return error
			if err != nil {
				_ = multierr.Append(errList, err)
			}
		} else {
			// if there is no local objects, return error
			_ = multierr.Append(errList, errors.Errorf("Failed to get local objects"))
		}

	}

	if errList != nil {
		log.Errorf(errList, "Failed to get local objects")
	}
	locations := b.GetLocationsByLocalObject(localObjects)
	nchunks, nodes := b.GetObjectInfo(locations, len(localObjects), b.replica)

	b.locations = locations
	b.nchunks = nchunks
	b.nodes = nodes
	b.jobGlobalObjectIDs = make(map[string][]string)
	// setup jobGlobalObjectIDs
	for _, o := range globalObjects {
		b.jobGlobalObjectIDs[(*o).Labels[labels.VineyardObjectJobLabel]] = append(
			b.jobGlobalObjectIDs[(*o).Labels[labels.VineyardObjectJobLabel]],
			(*o).Spec.ObjectID)
	}

	return b
}

func (b *BestEffortStrategy) TrackingChunksByAPI(deploymentName, namespace string) *BestEffortStrategy {
	err := b.ConvertOutputToObjectInfo(deploymentName, namespace)
	if err != nil {
		log.Errorf(err, "Failed to convert output to object info")
	}
	return b
}

func (b *BestEffortStrategy) BuildLsMetadataCmd(deploymentName, namespace string) *cobra.Command {
	cmd := ctlclient.NewLsMetadatasCmd()
	_ = cmd.Flags().Set("limit", "100000")
	_ = cmd.Flags().Set("namespace", namespace)
	_ = cmd.Flags().Set("deployment-name", deploymentName)
	_ = cmd.Flags().Set("format", "json")
	return cmd
}

func (b *BestEffortStrategy) BuildGetClusterInfoCmd(deploymentName, namespace string) *cobra.Command {
	cmd := ctlclient.NewGetClusterInfoCmd()
	_ = cmd.Flags().Set("namespace", namespace)
	_ = cmd.Flags().Set("deployment-name", deploymentName)
	_ = cmd.Flags().Set("format", "json")
	return cmd
}

// CaptureCmdOutput captures the output of command
func (b *BestEffortStrategy) CaptureLsMetadatasOutput(cmd *cobra.Command) []byte {
	// Create a buffer to capture the output
	rescueStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		log.Errorf(err, "Failed to create pipe")
	}

	cmd.Parent().PersistentPreRun(cmd, []string{})
	cmd.Run(cmd, []string{}) // this gets captured

	os.Stdout = w
	ctlclient.Output.Print()

	w.Close()
	out, err := io.ReadAll(r)
	if err != nil {
		log.Errorf(err, "Failed to read from buffer")
	}
	os.Stdout = rescueStdout

	return out
}

// ConvertOutputToObjectInfo converts the output of "vineyardctl ls metadatas" to object info.
func (b *BestEffortStrategy) ConvertOutputToObjectInfo(deploymentName, namespace string) error {
	// get the output of "vineyardctl ls metadatas"
	var metadataResult map[string]interface{}
	metaOutput := b.CaptureLsMetadatasOutput(b.BuildLsMetadataCmd(deploymentName, namespace))
	if err := json.Unmarshal(metaOutput, &metadataResult); err != nil {
		return errors.Errorf("Failed to unmarshal metadata output: %v", err)
	}

	// get the output of "vineyardctl get cluster-info"
	var clusterInfoResult map[string]interface{}
	clusterInfoOutput := b.CaptureLsMetadatasOutput(b.BuildGetClusterInfoCmd(deploymentName, namespace))
	if err := json.Unmarshal(clusterInfoOutput, &clusterInfoResult); err != nil {
		return errors.Errorf("Failed to unmarshal cluster-info output: %v", err)
	}

	requiredJobs := make(map[string]bool)
	for _, n := range b.required {
		requiredJobs[n] = true
	}

	// get instances -> Nodename
	instanceNodename := make(map[string]string)
	for k, v := range clusterInfoResult {
		instance := strings.Trim(k, "i")
		instanceNodename[instance] = v.(map[string]interface{})["nodename"].(string)
	}

	// jobname -> globalobject id
	jobGlobalObjectIDs := make(map[string][]string)
	// instance -> localobject id
	locations := make(map[string][]string)
	localObjectSum := 0

	for k, v := range metadataResult {
		if strings.Contains(k, "o") && v.(map[string]interface{})["global"] != nil {
			if v.(map[string]interface{})["global"].(bool) {
				jobName := v.(map[string]interface{})["JOB_NAME"].(string)
				jobGlobalObjectIDs[jobName] = append(jobGlobalObjectIDs[jobName], k)
				if requiredJobs[jobName] {
					// get all elements of the global object
					size := v.(map[string]interface{})["__elements_-size"].(float64)
					for i := 0; i < int(size); i++ {
						// get the localobect id
						localObjectID := v.(map[string]interface{})["__elements_-"+strconv.Itoa(i)].(map[string]interface{})["id"].(string)
						instanceID := v.(map[string]interface{})["instance_id"].(float64)
						nodeName := instanceNodename[strconv.Itoa(int(instanceID))]
						locations[nodeName] = append(locations[nodeName], localObjectID)
						localObjectSum++
					}
				}
			}
		}
	}
	nchunks, nodes := b.GetObjectInfo(locations, localObjectSum, b.replica)
	b.locations = locations
	b.nchunks = nchunks
	b.nodes = nodes
	b.jobGlobalObjectIDs = jobGlobalObjectIDs
	return nil
}

// GetLocalObjectsBySignatures returns the local objects by the given signatures.
func (b *BestEffortStrategy) GetLocalObjectsBySignatures(
	signatures []string,
) ([]*v1alpha1.LocalObject, error) {
	objects := make([]*v1alpha1.LocalObject, 0)
	for _, sig := range signatures {
		localObjects := &v1alpha1.LocalObjectList{}
		if err := b.List(context.TODO(), localObjects, client.MatchingLabels{
			"k8s.v6d.io/signature": sig,
		}); err != nil {
			return nil, err
		}
		for _, localObject := range localObjects.Items {
			objects = append(objects, &localObject)
		}
	}

	return objects, nil
}

func (b *BestEffortStrategy) GetLocationsByLocalObject(localObjects []*v1alpha1.LocalObject) map[string][]string {
	locations := make(map[string][]string)
	for _, localObject := range localObjects {
		host := localObject.Spec.Hostname
		if _, ok := locations[host]; !ok {
			locations[host] = make([]string, 0)
		}
		locations[host] = append(locations[host], localObject.Spec.ObjectID)
	}
	return locations
}

// GetObjectInfo returns the local object info including the locations and average number of chunks per node.
func (b *BestEffortStrategy) GetObjectInfo(
	locations map[string][]string,
	localObjectSum int,
	replica int,
) (int, []string) {
	// total frags
	totalfrags := localObjectSum
	// frags for per pod
	nchunks := totalfrags / replica
	if totalfrags%replica != 0 {
		nchunks++
	}

	// find the node
	nodes := make([]string, 0)
	for k := range locations {
		nodes = append(nodes, k)
	}
	sort.Strings(nodes)
	return nchunks, nodes
}

// GetGlobalObjectsByID returns the global objects by the given jobname.
func (b *BestEffortStrategy) GetGlobalObjectsByID(
	jobNames []string,
) ([]*v1alpha1.GlobalObject, error) {
	requiredJobs := make(map[string]bool)
	for _, n := range jobNames {
		requiredJobs[n] = true
	}
	objects := []*v1alpha1.GlobalObject{}
	globalObjects := &v1alpha1.GlobalObjectList{}
	if err := b.List(context.TODO(), globalObjects); err != nil {
		return nil, err
	}
	for i, obj := range globalObjects.Items {
		if jobname, exist := obj.Labels[labels.VineyardObjectJobLabel]; exist && requiredJobs[jobname] {
			objects = append(objects, &globalObjects.Items[i])
		}
	}

	return objects, nil
}

// GetLocalObjectsByID returns the local objects by the given jobname.
func (b *BestEffortStrategy) GetLocalObjectsByID(
	jobNames []string,
) ([]*v1alpha1.LocalObject, error) {
	requiredJobs := make(map[string]bool)
	for _, n := range jobNames {
		requiredJobs[n] = true
	}
	objects := []*v1alpha1.LocalObject{}
	localObjects := &v1alpha1.LocalObjectList{}
	if err := b.List(context.TODO(), localObjects); err != nil {
		return nil, err
	}
	for i, obj := range localObjects.Items {
		if jobname, exist := obj.Labels[labels.VineyardObjectJobLabel]; exist && requiredJobs[jobname] {
			objects = append(objects, &localObjects.Items[i])
		}
	}

	return objects, nil
}

// CreateConfigmapForID creates a configmap for the object id and the nodes.
func (b *BestEffortStrategy) CreateConfigmapForID(
	jobname []string,
	locations map[string][]string,
	jobGlobalObjectIDs map[string][]string,
) error {
	for i := range jobname {
		configmap := &v1.ConfigMap{}
		err := b.Get(
			context.TODO(),
			client.ObjectKey{Namespace: b.namespace, Name: jobname[i]},
			configmap,
		)
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		// the configmap doesn't exist
		if apierrors.IsNotFound(err) {
			data := make(map[string]string)
			for nodeName, nodeObjs := range locations {
				data[nodeName] = strings.Join(nodeObjs, ",")
			}
			// get all global objects produced by the required job
			data[jobname[i]] = strings.Join(jobGlobalObjectIDs[jobname[i]], ",")
			cm := v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobname[i],
					Namespace: b.namespace,
				},
				Data: data,
			}
			cm.OwnerReferences = *b.ownerReference
			if err := b.Create(context.TODO(), &cm); err != nil {
				return err
			}
		}
	}

	return nil
}

// Compute return the target node for the given rank.
func (b *BestEffortStrategy) Compute(rank int) (string, error) {
	target := ""

	cnt := 0
	for _, node := range b.nodes {
		localfrags := len(b.locations[node])
		if cnt+localfrags >= (b.nchunks*rank + (b.nchunks+1)/2) {
			target = node
			break
		}
		cnt += localfrags
	}

	// create configmap for each job
	if err := b.CreateConfigmapForID(b.required, b.locations, b.jobGlobalObjectIDs); err != nil {
		return target, err
	}

	return target, nil
}
