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

// Package schedule contains the schedule command of vineyard operator
package schedule

import "github.com/spf13/cobra"

var scheduleCmd = &cobra.Command{
	Use:   "schedule",
	Short: "schedule return a nodeName for the workload to co-allocate with vineyard cluster",
	Long: `schedule return a nodeName for the workload to co-allocate with vineyard cluster.
For example:

# Schedule a workload to a vineyard cluster
# it will return a nodeName which is the node that 
# the workload should be scheduled to
vineyardctl schedule --workload=workloadName`,
}

func NewScheduleCmd() *cobra.Command {
	return scheduleCmd
}

func init() {
	scheduleCmd.AddCommand(NewScheduleWorkloadCmd())
}
