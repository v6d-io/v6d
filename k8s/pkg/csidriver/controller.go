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
	"sync"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/v6d-io/v6d/k8s/pkg/log"
)

var (
	controllerCaps = []csi.ControllerServiceCapability_RPC_Type{
		csi.ControllerServiceCapability_RPC_CREATE_DELETE_VOLUME,
		csi.ControllerServiceCapability_RPC_PUBLISH_UNPUBLISH_VOLUME,
	}
	VineyardSocketPrefix = "/var/run/vineyard-kubernetes"
	VineyardSocket       = "vineyard.sock"

	// VineyardNamespaceKey is the key for the vineyard name in the volume context
	VineyardNamespaceKey = "k8s.v6d.io/vineyard/namespace"

	// VineyardNameKey is the key for the vineyard name in the volume context
	VineyardNameKey = "k8s.v6d.io/vineyard/name"
)

type VineyardCSI struct {
	mutex  sync.Mutex
	state  *VolumeStates
	nodeID string
}

func NewVineyardCSI(stateFilePath string, nodeID string) *VineyardCSI {
	volumeStates, err := NewVolumeStates(stateFilePath)
	if err != nil {
		log.Fatalf(err, "failed to create volume states")
	}
	return &VineyardCSI{
		mutex:  sync.Mutex{},
		state:  volumeStates,
		nodeID: nodeID,
	}
}

func (vc *VineyardCSI) ControllerGetCapabilities(ctx context.Context,
	req *csi.ControllerGetCapabilitiesRequest) (*csi.ControllerGetCapabilitiesResponse, error) {
	log.Infof("ControllerGetCapabilities: called with args %+v", *req)
	caps := []*csi.ControllerServiceCapability{}
	for _, cap := range controllerCaps {
		c := &csi.ControllerServiceCapability{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: cap,
				},
			},
		}
		caps = append(caps, c)
	}
	return &csi.ControllerGetCapabilitiesResponse{Capabilities: caps}, nil
}

func (vc *VineyardCSI) CreateVolume(ctx context.Context, req *csi.CreateVolumeRequest) (*csi.CreateVolumeResponse, error) {
	log.Infof("CreateVolume: called with args %+v", *req)
	// check arguments
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "CreateVolume Name must be provided")
	}

	if req.VolumeCapabilities == nil || len(req.VolumeCapabilities) == 0 {
		return nil, status.Error(codes.InvalidArgument, "CreateVolume Volume capabilities must be provided")
	}
	vc.mutex.Lock()
	defer vc.mutex.Unlock()

	capacity := req.GetCapacityRange().GetRequiredBytes()
	// check whether the volume exists
	if exVol, err := vc.state.GetVolumeByName(req.Name); err == nil {
		if exVol.VolSize != capacity {
			return nil, status.Errorf(codes.AlreadyExists, "Volume with the same name: %s but with different size already exist", req.GetName())
		}

		// volume already exists, return the volume
		return &csi.CreateVolumeResponse{
			Volume: &csi.Volume{
				VolumeId:      exVol.VolID,
				VolumeContext: req.GetParameters(),
			},
		}, nil
	}
	// create the uuid for the volume
	uuid, err := uuid.NewUUID()
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	volumeID := uuid.String()
	log.Infof("create volumeID: %s", volumeID)

	volumeContext := req.GetParameters()
	if len(volumeContext) == 0 {
		volumeContext = make(map[string]string)
	}
	volumeContext["volume_id"] = volumeID

	// add volume to the state
	volume := Volume{
		VolID:   volumeID,
		VolName: req.Name,
		VolSize: capacity,
	}
	if err := vc.state.AddVolume(volume); err != nil {
		return nil, err
	}

	return &csi.CreateVolumeResponse{
		Volume: &csi.Volume{
			VolumeId:      volumeID,
			VolumeContext: volumeContext,
		},
	}, nil
}

func (vc *VineyardCSI) DeleteVolume(ctx context.Context, req *csi.DeleteVolumeRequest) (*csi.DeleteVolumeResponse, error) {
	log.Infof("DeleteVolume: called with args: %+v", *req)
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "DeleteVolume Volume ID must be provided")
	}

	if err := vc.state.DeleteVolume(req.VolumeId); err != nil {
		return nil, err
	}
	return &csi.DeleteVolumeResponse{}, nil
}

func (vc *VineyardCSI) ControllerGetVolume(ctx context.Context, req *csi.ControllerGetVolumeRequest) (*csi.ControllerGetVolumeResponse, error) {
	log.Infof("ControllerGetVolume: called with args: %+v", *req)
	return &csi.ControllerGetVolumeResponse{}, nil
}

func (vc *VineyardCSI) ControllerPublishVolume(ctx context.Context, req *csi.ControllerPublishVolumeRequest) (*csi.ControllerPublishVolumeResponse, error) {
	log.Infof("ControllerPublishVolume: called with args %+v", *req)
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "ControllerPublishVolume Volume ID must be provided")
	}

	if req.NodeId == "" {
		return nil, status.Error(codes.InvalidArgument, "ControllerPublishVolume Node ID must be provided")
	}

	if req.VolumeCapability == nil {
		return nil, status.Error(codes.InvalidArgument, "ControllerPublishVolume Volume capability must be provided")
	}

	vc.mutex.Lock()
	defer vc.mutex.Unlock()

	vol, err := vc.state.GetVolumeByID(req.GetVolumeId())
	if err != nil {
		return nil, err
	}

	if vol.Attched {
		if req.GetReadonly() != vol.ReadOnlyAttach {
			return nil, status.Error(codes.AlreadyExists, "Volume published but has incompatible readonly flag")
		}

		return &csi.ControllerPublishVolumeResponse{
			PublishContext: map[string]string{},
		}, nil
	}

	vol.Attched = true
	vol.ReadOnlyAttach = req.GetReadonly()

	if err := vc.state.AddVolume(vol); err != nil {
		return nil, err
	}
	return &csi.ControllerPublishVolumeResponse{}, nil
}

func (vc *VineyardCSI) ControllerUnpublishVolume(ctx context.Context,
	req *csi.ControllerUnpublishVolumeRequest) (*csi.ControllerUnpublishVolumeResponse, error) {
	log.Infof("ControllerUnpublishVolume: called with args %+v", *req)
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "ControllerPublishVolume Volume ID must be provided")
	}
	return &csi.ControllerUnpublishVolumeResponse{}, nil
}

func (vc *VineyardCSI) ValidateVolumeCapabilities(ctx context.Context,
	req *csi.ValidateVolumeCapabilitiesRequest) (*csi.ValidateVolumeCapabilitiesResponse, error) {
	if req.VolumeId == "" {
		return nil, status.Error(codes.InvalidArgument, "ValidateVolumeCapabilities Volume ID must be provided")
	}

	if req.VolumeCapabilities == nil {
		return nil, status.Error(codes.InvalidArgument, "ValidateVolumeCapabilities Volume Capabilities must be provided")
	}
	vc.mutex.Lock()
	defer vc.mutex.Unlock()

	if _, err := vc.state.GetVolumeByID(req.GetVolumeId()); err != nil {
		return nil, err
	}

	for _, cap := range req.GetVolumeCapabilities() {
		if cap.GetMount() == nil && cap.GetBlock() == nil {
			return nil, status.Error(codes.InvalidArgument, "cannot have both mount and block access type be undefined")
		}
	}

	return &csi.ValidateVolumeCapabilitiesResponse{
		Confirmed: &csi.ValidateVolumeCapabilitiesResponse_Confirmed{
			VolumeContext:      req.GetVolumeContext(),
			VolumeCapabilities: req.GetVolumeCapabilities(),
			Parameters:         req.GetParameters(),
		},
	}, nil

}

func (vc *VineyardCSI) ListVolumes(ctx context.Context, req *csi.ListVolumesRequest) (*csi.ListVolumesResponse, error) {
	return nil, status.Error(codes.Unimplemented, "")
}

func (vc *VineyardCSI) GetCapacity(ctx context.Context, req *csi.GetCapacityRequest) (*csi.GetCapacityResponse, error) {
	return nil, status.Error(codes.Unimplemented, "")
}

func (vc *VineyardCSI) CreateSnapshot(ctx context.Context, req *csi.CreateSnapshotRequest) (*csi.CreateSnapshotResponse, error) {
	if req.GetName() == "" {
		return nil, status.Error(codes.InvalidArgument, "CreateSnapshot Name must be provided")
	}

	if req.GetSourceVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "CreateSnapshot Source Volume ID must be provided")
	}

	return nil, status.Error(codes.Unimplemented, "")
}

func (vc *VineyardCSI) DeleteSnapshot(ctx context.Context, req *csi.DeleteSnapshotRequest) (*csi.DeleteSnapshotResponse, error) {
	return nil, status.Error(codes.Unimplemented, "")
}

func (vc *VineyardCSI) ListSnapshots(ctx context.Context, req *csi.ListSnapshotsRequest) (*csi.ListSnapshotsResponse, error) {
	return nil, status.Error(codes.Unimplemented, "")
}

func (vc *VineyardCSI) ControllerExpandVolume(ctx context.Context, req *csi.ControllerExpandVolumeRequest) (*csi.ControllerExpandVolumeResponse, error) {
	return nil, status.Error(codes.Unimplemented, "")
}
