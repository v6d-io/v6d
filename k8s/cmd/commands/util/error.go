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
package util

import "github.com/pkg/errors"

// ErrorCollector collects errors and returns them as a single error.
type ErrorCollector struct {
	errlists []error
}

// Add a new error to the collector.
func (e *ErrorCollector) Add(err error) {
	if err != nil {
		e.errlists = append(e.errlists, err)
	}
}

// Error returns the collected errors as a single error.
func (e *ErrorCollector) Error() error {
	errStr := ""
	if len(e.errlists) == 0 {
		return nil
	}

	for _, e := range e.errlists {
		errStr += e.Error() + ";"
	}
	return errors.New(errStr)
}
