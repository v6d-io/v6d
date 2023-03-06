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

import (
	"io"
	"os"

	"github.com/pkg/errors"
)

// ReadFromFile reads the file and returns the content as a string.
func ReadFromFile(path string) (string, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	return string(contents), nil
}

// ReadFromStdin read the stdin to string
func ReadFromStdin(args []string) (string, error) {
	// Check if the input is coming from '-'
	if len(args) > 0 && args[0] == "-" {
		// Read all the bytes piped in from stdin
		r, err := io.ReadAll(os.Stdin)
		if err != nil {
			return "", errors.Wrap(err, "failed to read from stdin")
		}
		return string(r), nil
	}
	return "", nil
}

// ReadJsonFromStdin read the stdin to json string
func ReadJsonFromStdin(args []string) (string, error) {
	// Check if the input is coming from '-'
	input, err := ReadFromStdin(args)
	if err != nil {
		return "", err
	}

	j, err := ConvertToJson(input)
	if err != nil {
		return "", errors.Wrap(err, "failed to convert to json")
	}
	return j, nil
}
