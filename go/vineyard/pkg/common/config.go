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

package common

import "fmt"

const (
	VINEYARD_VERSION_MAJOR = 0
	VINEYARD_VERSION_MINOR = 7
	VINEYARD_VERSION_PATCH = 2

	VINEYARD_VERSION = ((VINEYARD_VERSION_MAJOR*1000)+VINEYARD_VERSION_MINOR)*1000 +
		VINEYARD_VERSION_PATCH
)

var VINEYARD_VERSION_STRING = fmt.Sprintf(
	"%d.%d.%d",
	VINEYARD_VERSION_MAJOR,
	VINEYARD_VERSION_MINOR,
	VINEYARD_VERSION_PATCH,
)
