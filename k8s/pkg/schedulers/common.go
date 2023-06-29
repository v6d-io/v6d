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
	"sort"
	"strings"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/v6d-io/v6d/k8s/pkg/config/annotations"
	"github.com/v6d-io/v6d/k8s/pkg/config/labels"
)

// GetVineyarddNodes returns all node names of vineyardd pods.
func GetVineyarddNodes(c client.Client, jobLabels map[string]string) ([]string, error) {
	nodes := []string{}

	vineyarddName, exist := jobLabels[labels.VineyarddName]
	if !exist {
		return nodes, errors.Errorf("Failed to get vineyardd name")
	}
	vineyarddNamespace, exist := jobLabels[labels.VineyarddNamespace]
	if !exist {
		return nodes, errors.Errorf("Failed to get vineyardd name")
	}

	podList := v1.PodList{}
	option := &client.ListOptions{
		LabelSelector: apilabels.SelectorFromSet(apilabels.Set{
			"app.kubernetes.io/name":     vineyarddName,
			"app.kubernetes.io/instance": vineyarddNamespace + "-" + vineyarddName,
		}),
	}

	if err := c.List(context.TODO(), &podList, option); err != nil {
		return nodes, errors.Wrap(err, "Failed to list all pods with the specific label")
	}

	for _, pod := range podList.Items {
		nodes = append(nodes, pod.Spec.NodeName)
	}
	sort.Strings(nodes)
	return nodes, nil
}

// GetRequiredJob get all required jobs name that separated by ',' from annotations
func GetRequiredJob(anno map[string]string) []string {
	requiredJobs, exists := anno[annotations.VineyardJobRequired]
	if !exists {
		return []string{}
	}
	if requiredJobs == "" {
		return []string{}
	}

	return strings.Split(requiredJobs, ",")
}
