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

import (
	"fmt"

	"github.com/pkg/errors"
)

const (
	KOK                       = 0
	KInvalid                  = 1
	KKeyError                 = 2
	KTypeError                = 3
	KIOError                  = 4
	KEndOfFile                = 5
	KNotImplemented           = 6
	KAssertionFailed          = 7
	KUserInputError           = 8
	KObjectExists             = 11
	KObjectNotExists          = 12
	KObjectSealed             = 13
	KObjectNotSealed          = 14
	KObjectIsBlob             = 15
	KMetaTreeInvalid          = 21
	KMetaTreeTypeInvalid      = 22
	KMetaTreeTypeNotExists    = 23
	KMetaTreeNameInvalid      = 24
	KMetaTreeNameNotExists    = 25
	KMetaTreeLinKInvalid      = 26
	KMetaTreeSubtreeNotExists = 27
	KVineyardServerNotReady   = 31
	KArrowError               = 32
	KConnectionFailed         = 33
	KConnectionError          = 34
	KEtcdError                = 35
	KNotEnoughMemory          = 41
	KStreamDrained            = 42
	KStreamFailed             = 43
	KInvalidStreamState       = 44
	KStreamOpened             = 45
	KGlobalObjectInvalid      = 51
	KUnKnownError             = 255
)

var ErrCodes map[int]string

func init() {
	ErrCodes = make(map[int]string)

	ErrCodes[0] = "OK"
	ErrCodes[1] = "Invalid"
	ErrCodes[2] = "KeyError"
	ErrCodes[3] = "TypeError"
	ErrCodes[4] = "IOError"
	ErrCodes[5] = "EndOfFile"
	ErrCodes[6] = "NotImplemented"
	ErrCodes[7] = "AssertionFailed"
	ErrCodes[8] = "UserInputError"
	ErrCodes[11] = "ObjectExists"
	ErrCodes[12] = "ObjectNotExists"
	ErrCodes[13] = "ObjectSealed"
	ErrCodes[14] = "ObjectNotSealed"
	ErrCodes[15] = "ObjectIsBlob"
	ErrCodes[21] = "MetaTreeInvalid"
	ErrCodes[22] = "MetaTreeTypeInvalid"
	ErrCodes[23] = "MetaTreeTypeNotExists"
	ErrCodes[24] = "MetaTreeNameInvalid"
	ErrCodes[25] = "MetaTreeNameNotExists"
	ErrCodes[26] = "MetaTreeLinKInvalid"
	ErrCodes[27] = "MetaTreeSubtreeNotExists"
	ErrCodes[31] = "VineyardServerNotReady"
	ErrCodes[32] = "ArrowError"
	ErrCodes[33] = "ConnectionFailed"
	ErrCodes[34] = "ConnectionError"
	ErrCodes[35] = "EtcdError"
	ErrCodes[41] = "NotEnoughMemory"
	ErrCodes[42] = "StreamDrained"
	ErrCodes[43] = "StreamFailed"
	ErrCodes[44] = "InvalidStreamState"
	ErrCodes[45] = "StreamOpened"
	ErrCodes[51] = "GlobalObjectInvalid"
	ErrCodes[255] = "UnKnownError"
}

type Status struct {
	Code    int
	Message string
}

func (r *Status) Error() string {
	m := "UnknownError"
	if k, ok := ErrCodes[r.Code]; ok {
		m = k
	}
	return fmt.Sprintf("code: %v, message: %v: %+v", r.Code, m, r.Message)
}

func (r *Status) Wrap() error {
	return errors.WithStack(r)
}

func Error(code int, message string) error {
	err := &Status{code, message}
	return err.Wrap()
}

func ReplyTypeMismatch(t string) error {
	return Error(KInvalid, fmt.Sprintf("reply type mismatch, expect %v", t))
}

func NotConnected() error {
	return Error(KAssertionFailed, "client not connected")
}
