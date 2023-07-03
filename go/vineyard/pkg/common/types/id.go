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
	"fmt"
	"strconv"
	"time"
)

type ObjectID = uint64

func GenerateObjectID() ObjectID {
	// TODO: check c++ version's rdtsc instead of time.Now() in golang
	return ObjectID(0x7FFFFFFFFFFFFFFF & time.Now().Unix())
}

func ObjectIDToString(id ObjectID) string {
	return fmt.Sprintf("o%016x", id)
}

func ObjectIDFromString(id string) (ObjectID, error) {
	return strconv.ParseUint(id[1:], 16, 64)
}

func IsBlob(id ObjectID) bool {
	return id&0x8000000000000000 != 0
}

func InvalidObjectID() ObjectID {
	return 0xffffffffffffffff
}

func EmptyBlobID() ObjectID {
	return 0x8000000000000000
}

type Signature = uint64

func SignatureToString(sig Signature) string {
	return fmt.Sprintf("s%016x", sig)
}

func SignatureFromString(sig string) (Signature, error) {
	return strconv.ParseUint(sig[1:], 16, 64)
}

func InvalidSignature() Signature {
	return 0xffffffffffffffff
}

type InstanceID = uint64

func UnspecifiedInstanceID() InstanceID {
	return 0xffffffffffffffff
}

type SessionID = uint64

func SessionIDToString(sig SessionID) string {
	return fmt.Sprintf("S%016x", sig)
}

func SessionIDFromString(sig string) (SessionID, error) {
	return strconv.ParseUint(sig[1:], 16, 64)
}

func RootSessionID() SessionID {
	return 0
}
