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

// Package dryschedule contains the schedule command of vineyard operator
package dryschedule

import "github.com/spf13/cobra"

var dryscheduleCmd = &cobra.Command{
	Use:   "dryschedule",
	Short: "Dryschedule return a nodeName for the workload to co-allocation with vineyard cluster",
	Long: `Dryschedule return a nodeName for the workload to co-allocation with vineyard cluster.
For example:

# Dryschedule a workload to a vineyard cluster
# it will return a nodeName which is the node that 
# the workload should be scheduled to
vineyarctl dryschedule --workload=workloadName`,
	Run: func(cmd *cobra.Command, args []string) {
	},
}

func NewDryScheduleCmd() *cobra.Command {
	return dryscheduleCmd
}

func init() {
	dryscheduleCmd.AddCommand(NewScheduleWorkloadCmd())
}
