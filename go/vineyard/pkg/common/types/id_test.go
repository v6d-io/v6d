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

package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestObjectID(t *testing.T) {
	var s string = ObjectIDToString(1234)
	o, _ := ObjectIDFromString(s)
	assert.Equal(t, s, "o00000000000004d2")
	assert.Equal(t, o, uint64(1234))
}

func TestSignature(t *testing.T) {
	var s string = SignatureToString(1234)
	o, _ := SignatureFromString(s)
	assert.Equal(t, s, "s00000000000004d2")
	assert.Equal(t, o, uint64(1234))
}
