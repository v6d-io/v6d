/*
* Copyright 2020-2023 Alibaba Group Holding Limited.

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
package flags

import "github.com/spf13/cobra"

var (
	// Resource is the json string of kubernetes workload
	Resource string

	// the namespace of vineyard cluster
	VineyarddNamespace string

	// the output format for vineyardctl schedule workload command
	ScheduleOutputFormat string

	// the file path of workload
	WorkloadFile string
)

func ApplySchedulerWorkloadOpts(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&Resource, "resource", "", "",
		"the json string of kubernetes workload")
	cmd.Flags().StringVarP(&WorkloadFile, "file", "f", "", "the file path of workload")
	cmd.Flags().StringVarP(&ScheduleOutputFormat, "output", "o", "json",
		"the output format for vineyardctl schedule workload command")
	cmd.Flags().
		StringVarP(&VineyarddName, "vineyardd-name", "", "vineyardd-sample",
			"the namespace of vineyard cluster")
	cmd.Flags().
		StringVarP(&VineyarddNamespace, "vineyardd-namespace", "", "vineyard-system",
			"the namespace of vineyard cluster")
}
