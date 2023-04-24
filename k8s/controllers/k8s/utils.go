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

package k8s

import (
	"crypto/rand"
	"fmt"
	"strconv"
	"strings"

	"github.com/v6d-io/v6d/k8s/pkg/log"
)

func GenerateRandomName(length int) string {
	bs := make([]byte, length)
	if _, err := rand.Read(bs); err != nil {
		log.Fatal(err, "rand.Read failed")
	}
	return fmt.Sprintf("%x", bs)[:length]
}

// EtcdConfig holds all configuration about etcd
type EtcdConfig struct {
	Name      string
	Namespace string
	Rank      int
	Endpoints string
	Image     string
}

// NewEtcdConfig builds the etcd config.
func NewEtcdConfig(name string, namespace string,
	replicas int, image string,
) EtcdConfig {
	etcdConfig := EtcdConfig{}
	// the etcd is built in the vineyardd image
	etcdConfig.Name = name
	etcdConfig.Image = image
	etcdConfig.Namespace = namespace
	etcdEndpoints := make([]string, 0, replicas)
	for i := 0; i < replicas; i++ {
		etcdName := fmt.Sprintf("%v-etcd-%v", name, strconv.Itoa(i))
		etcdEndpoints = append(
			etcdEndpoints,
			fmt.Sprintf("%v=http://%v:2380", etcdName, etcdName),
		)
	}
	etcdConfig.Endpoints = strings.Join(etcdEndpoints, ",")
	return etcdConfig
}
