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
	KOK              = 0
	KInvalid         = 1
	KKeyError        = 2
	KTypeError       = 3
	KIOError         = 4
	KEndOfFile       = 5
	KNotImplemented  = 6
	KAssertionFailed = 7
	KUserInputError  = 8

	KObjectExists    = 11
	KObjectNotExists = 12
	KObjectSealed    = 13
	KObjectNotSealed = 14
	KObjectIsBlob    = 15

	KMetaTreeInvalid          = 21
	KMetaTreeTypeInvalid      = 22
	KMetaTreeTypeNotExists    = 23
	KMetaTreeNameInvalid      = 24
	KMetaTreeNameNotExists    = 25
	KMetaTreeLinKInvalid      = 26
	KMetaTreeSubtreeNotExists = 27

	KVineyardServerNotReady = 31
	KArrowError             = 32
	KConnectionFailed       = 33
	KConnectionError        = 34
	KEtcdError              = 35

	KNotEnoughMemory    = 41
	KStreamDrained      = 42
	KStreamFailed       = 43
	KInvalidStreamState = 44
	KStreamOpened       = 45

	KGlobalObjectInvalid = 51

	KUnKnownError = 255
)

type ReplyError struct {
	Code int
	Type string
	Err  error
}

func (r *ReplyError) Error() string {
	return "code:" + fmt.Sprintf("%v", r.Code) + " type:" + r.Type + " :" + r.Err.Error()
}
