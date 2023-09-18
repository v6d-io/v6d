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
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Volume struct {
	VolName        string
	VolID          string
	VolSize        int64
	VolPath        string
	NodeID         string
	ReadOnlyAttach bool
	Attched        bool
}

type VolumeStates struct {
	Volumes       []Volume
	StateFilePath string
}

func NewVolumeStates(statefilePath string) (*VolumeStates, error) {
	s := &VolumeStates{
		StateFilePath: statefilePath,
	}

	return s, s.restore()
}

func (s *VolumeStates) dump() error {
	data, err := json.Marshal(&s.Volumes)
	if err != nil {
		return status.Errorf(codes.Internal, "error encoding volumes and snapshots: %v", err)
	}
	if err := os.WriteFile(s.StateFilePath, data, 0600); err != nil {
		return status.Errorf(codes.Internal, "error writing state file: %v", err)
	}
	return nil
}

func (s *VolumeStates) restore() error {
	s.Volumes = nil

	data, err := os.ReadFile(s.StateFilePath)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// create the /csi directory if it does not exist
		if err := os.MkdirAll(filepath.Dir(s.StateFilePath), 0750); err != nil {
			return status.Errorf(codes.Internal, "error creating state file directory: %v", err)
		}
		// create the state file if it does not exist
		if _, err := os.Create(s.StateFilePath); err != nil {
			return status.Errorf(codes.Internal, "error creating state file: %v", err)
		}
		return nil
	case err != nil:
		return status.Errorf(codes.Internal, "error reading state file: %v", err)
	}
	if len(data) == 0 {
		s.Volumes = []Volume{}
		return nil
	}
	if err := json.Unmarshal(data, &s.Volumes); err != nil {
		return status.Errorf(codes.Internal, "error encoding volumes and snapshots from state file %q: %v", s.StateFilePath, err)
	}
	return nil
}

func (s *VolumeStates) GetVolumeByID(volID string) (Volume, error) {
	for _, volume := range s.Volumes {
		if volume.VolID == volID {
			return volume, nil
		}
	}
	return Volume{}, status.Errorf(codes.NotFound, "volume id %s does not exist in the volumes list", volID)
}

func (s *VolumeStates) GetVolumeByName(volName string) (Volume, error) {
	for _, volume := range s.Volumes {
		if volume.VolName == volName {
			return volume, nil
		}
	}
	return Volume{}, status.Errorf(codes.NotFound, "volume name %s does not exist in the volumes list", volName)
}

func (s *VolumeStates) AddVolume(v Volume) error {
	for i, volume := range s.Volumes {
		if volume.VolID == v.VolID {
			s.Volumes[i] = v
			return nil
		}
	}
	s.Volumes = append(s.Volumes, v)
	return s.dump()
}

func (s *VolumeStates) DeleteVolume(volID string) error {
	for i, volume := range s.Volumes {
		if volume.VolID == volID {
			s.Volumes = append(s.Volumes[:i], s.Volumes[i+1:]...)
			return s.dump()
		}
	}
	return nil
}
