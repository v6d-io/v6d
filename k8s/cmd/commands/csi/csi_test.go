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

// Package csi contains the start command of vineyard csi driver
package csi

import (
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/kubernetes-csi/csi-test/v4/pkg/sanity"
	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/pkg/csidriver"
)

func TestVineyardCSIDriver(t *testing.T) {
	// Setup the full driver and its environment
	csiSocket := "/tmp/csi.sock"
	csiEndpoint := "unix://" + csiSocket
	if err := os.Remove(csiSocket); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to remove socket file %s: %v", csiSocket, err)
		os.Exit(1)
	}

	flags.Endpoint = csiEndpoint
	flags.NodeID = "test-node-id"

	vineyardSocket := filepath.Join(csidriver.VineyardSocketPrefix, csidriver.VineyardSocket)
	if _, err := os.OpenFile(vineyardSocket, os.O_CREATE|os.O_RDONLY, 0666); err != nil {
		t.Errorf("failed to open vineyard socket file %s: %v", vineyardSocket, err)
	}

	// Create a channel to signal the goroutine to stop
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		csiCmd.Run(csiCmd, []string{})
	}()
	config := sanity.NewTestConfig()
	config.Address = csiEndpoint
	config.TargetPath = "/opt/target"
	config.StagingPath = "/opt/staging"

	sanity.Test(t, config)

	// Wait for a stop signal
	<-stop

	// Exit the program with a status code of 0 (success)
	os.Exit(0)
}
