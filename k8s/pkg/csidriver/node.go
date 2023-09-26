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
	"os"
	"path/filepath"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/pkg/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/mount-utils"

	"github.com/v6d-io/v6d/k8s/pkg/log"
)

// MaxVolumesPerNode is the maximum number of volumes supported per node.
const MaxVolumesPerNode = 1000

func (vc *VineyardCSI) NodeStageVolume(ctx context.Context, req *csi.NodeStageVolumeRequest) (*csi.NodeStageVolumeResponse, error) {
	log.Infof("NodeStageVolume: called with args %+v", *req)
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeStageVolume Volume ID must be provided")
	}

	if req.StagingTargetPath == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeStageVolume Staging Target Path must be provided")
	}

	if req.VolumeCapability == nil {
		return nil, status.Error(codes.InvalidArgument, "NodeStageVolume Volume Capability must be provided")
	}
	return &csi.NodeStageVolumeResponse{}, nil
}

func (vc *VineyardCSI) NodeUnstageVolume(ctx context.Context, req *csi.NodeUnstageVolumeRequest) (*csi.NodeUnstageVolumeResponse, error) {
	log.Infof("NodeUnstageVolume: called with args %+v", *req)
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeUnstageVolume Volume ID must be provided")
	}

	if req.StagingTargetPath == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeUnstageVolume Staging Target Path must be provided")
	}

	return &csi.NodeUnstageVolumeResponse{}, nil
}

func (vc *VineyardCSI) NodePublishVolume(ctx context.Context, req *csi.NodePublishVolumeRequest) (*csi.NodePublishVolumeResponse, error) {
	log.Infof("NodePublishVolume: called with args %+v", *req)

	// Check arguments
	if req.GetVolumeCapability() == nil {
		return nil, status.Error(codes.InvalidArgument, "Volume capability missing in request")
	}
	if len(req.GetVolumeId()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume ID missing in request")
	}
	if len(req.GetTargetPath()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Target path missing in request")
	}
	log.Infof("args check success")

	volumeContext := req.VolumeContext
	vineyardNamespace, ok := volumeContext[VineyardNamespaceKey]
	if !ok {
		return nil, status.Error(codes.InvalidArgument, fmt.Sprintf("vineyard namespace %s not found", VineyardNamespaceKey))
	}
	vineyardName, ok := volumeContext[VineyardNameKey]
	if !ok {
		return nil, status.Error(codes.InvalidArgument, fmt.Sprintf("vineyard name %s not found", VineyardNameKey))
	}

	vineyardSocketDir := filepath.Join(VineyardSocketPrefix, vineyardNamespace, vineyardName)
	socket := filepath.Join(vineyardSocketDir, VineyardSocket)
	if _, err := os.Stat(socket); err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("vineyard socket %s not found: %v", socket, err))
	}

	// check if target directory exists, create if not
	targetPath := req.GetTargetPath()
	log.Infof("target path: %s", targetPath)
	_, err := os.Stat(targetPath)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(targetPath, 0750); err != nil {
			return nil, status.Error(codes.Internal, fmt.Sprintf("create directory path {%s} error: %v", targetPath, err))
		}
	} else if err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("get target path {%s} failed: %v", targetPath, err))
	}

	mounter := mount.New("")
	options := []string{"bind"}
	if err := mounter.Mount(vineyardSocketDir, targetPath, "", options); err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("mount %s to %s failed: %v", vineyardSocketDir, targetPath, err))
	}
	log.Infof("mount %s to %s success", vineyardSocketDir, req.TargetPath)

	return &csi.NodePublishVolumeResponse{}, nil
}

func (vc *VineyardCSI) NodeUnpublishVolume(ctx context.Context, req *csi.NodeUnpublishVolumeRequest) (*csi.NodeUnpublishVolumeResponse, error) {
	log.Infof("NodeUnpublishVolume: called with args %+v", *req)
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeUnpublishVolume Volume ID must be provided")
	}

	if req.TargetPath == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeUnpublishVolume Target Path must be provided")
	}

	targetPath := req.GetTargetPath()
	if err := mount.CleanupMountPoint(targetPath, mount.New(""), true); err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("unmount %s failed: %v", targetPath, err))
	}
	log.Infof("unmount %s success", targetPath)

	return &csi.NodeUnpublishVolumeResponse{}, nil
}

func (vc *VineyardCSI) NodeGetInfo(ctx context.Context, req *csi.NodeGetInfoRequest) (*csi.NodeGetInfoResponse, error) {
	log.Infof("NodeGetInfo: called with args %+v", *req)

	return &csi.NodeGetInfoResponse{
		NodeId:            vc.nodeID,
		MaxVolumesPerNode: MaxVolumesPerNode,
		AccessibleTopology: &csi.Topology{
			Segments: map[string]string{
				"kubernetes.io/hostname": vc.nodeID,
			},
		},
	}, nil
}

func (vc *VineyardCSI) NodeGetCapabilities(ctx context.Context, req *csi.NodeGetCapabilitiesRequest) (*csi.NodeGetCapabilitiesResponse, error) {
	log.Infof("NodeGetCapabilities: called with args %+v", *req)

	return &csi.NodeGetCapabilitiesResponse{
		Capabilities: []*csi.NodeServiceCapability{
			{
				Type: &csi.NodeServiceCapability_Rpc{
					Rpc: &csi.NodeServiceCapability_RPC{
						Type: csi.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME,
					},
				},
			},
		},
	}, nil
}

func (vc *VineyardCSI) NodeGetVolumeStats(ctx context.Context, req *csi.NodeGetVolumeStatsRequest) (*csi.NodeGetVolumeStatsResponse, error) {
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "NodeGetVolumeStats Volume ID must be provided")
	}
	return nil, status.Error(codes.Unimplemented, "")
}

func (vc *VineyardCSI) NodeExpandVolume(ctx context.Context, req *csi.NodeExpandVolumeRequest) (*csi.NodeExpandVolumeResponse, error) {
	if len(req.GetVolumeId()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "NodeExpandVolume volume ID not provided")
	}

	if len(req.GetVolumePath()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "NodeExpandVolume volume path not provided")
	}
	return nil, status.Error(codes.Unimplemented, "")
}
