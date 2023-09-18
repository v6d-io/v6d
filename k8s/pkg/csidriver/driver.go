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
package csidriver

import (
	"context"
	"fmt"
	"net"
	"os"
	"strings"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/kubernetes-csi/csi-lib-utils/protosanitizer"
	"github.com/pkg/errors"
	"google.golang.org/grpc"

	"github.com/v6d-io/v6d/k8s/cmd/commands/flags"
	"github.com/v6d-io/v6d/k8s/pkg/log"
)

type Driver struct {
	nodeID   string
	endpoint string
}

const (
	version    = "0.1.0"
	driverName = "csi.vineyard.v6d.io"
)

func NewDriver(nodeID, endpoint string) *Driver {
	log.Infof("Driver: %v version: %v", driverName, version)

	n := &Driver{
		nodeID:   nodeID,
		endpoint: endpoint,
	}

	return n
}

func (d *Driver) Run() {

	vineyardCSI := NewVineyardCSI(flags.StateFilePath, d.nodeID)
	identity := NewIdentityServer()

	opts := []grpc.ServerOption{}
	if flags.Verbose {
		opts = append(opts, grpc.UnaryInterceptor(logGRPC))
	}

	srv := grpc.NewServer(opts...)

	csi.RegisterControllerServer(srv, vineyardCSI)
	csi.RegisterIdentityServer(srv, identity)
	csi.RegisterNodeServer(srv, vineyardCSI)

	proto, addr, err := ParseEndpoint(d.endpoint)
	log.Infof("protocol: %s,addr: %s", proto, addr)
	if err != nil {
		log.Fatalf(err, "Invalid endpoint: %v", d.endpoint)
	}

	if proto == "unix" {
		addr = "/" + addr
		if err := os.Remove(addr); err != nil && !os.IsNotExist(err) {
			log.Fatalf(err, "Failed to remove %s, error: %s", addr, err.Error())
		}
	}

	listener, err := net.Listen(proto, addr)
	if err != nil {
		log.Fatalf(err, "Failed to listen")
	}

	if err := srv.Serve(listener); err != nil {
		log.Fatalf(err, "Failed to serve")
	}
}

func logGRPC(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	log.Infof("GRPC call: %s", info.FullMethod)
	log.Infof("GRPC request: %s", protosanitizer.StripSecrets(req))

	resp, err := handler(ctx, req)
	if err != nil {
		log.Errorf(err, "GRPC error: %s", err.Error())
	} else {
		log.Infof("GRPC response: %s", protosanitizer.StripSecrets(resp))
	}
	return resp, err
}

func ParseEndpoint(ep string) (string, string, error) {
	if strings.HasPrefix(strings.ToLower(ep), "unix://") || strings.HasPrefix(strings.ToLower(ep), "tcp://") {
		s := strings.SplitN(ep, "://", 2)
		if s[1] != "" {
			return s[0], s[1], nil
		}
	}
	return "", "", errors.Wrap(fmt.Errorf("invalid endpoint: %s", ep), "parse endpoint")
}
