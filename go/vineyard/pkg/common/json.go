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
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

func ParseJson(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()

	if err := dec.Decode(v); err != nil {
		return err
	}
	return nil
}

func ParseJsonString(data string, v any) error {
	dec := json.NewDecoder(strings.NewReader(data))
	dec.UseNumber()

	if err := dec.Decode(v); err != nil {
		return err
	}
	return nil
}

func GetInt64(data map[string]any, key string) (int64, error) {
	if item, ok := data[key]; ok {
		if n, ok := item.(json.Number); ok {
			if v, err := strconv.ParseInt(string(n), 10, 64); err == nil {
				return v, nil
			}
			return 0, errors.Errorf("Key '%s' is not a number type", key)
		}
		return 0, errors.Errorf("Key '%s' is not a number type", key)
	}
	return 0, errors.Errorf("Key '%s' not found", key)
}

func GetUint64(data map[string]any, key string) (uint64, error) {
	if item, ok := data[key]; ok {
		if n, ok := item.(json.Number); ok {
			if v, err := strconv.ParseUint(string(n), 10, 64); err == nil {
				return v, nil
			}
			return 0, errors.Errorf("Key '%s' is not a number type", key)
		}
		return 0, errors.Errorf("Key '%s' is not a number type", key)
	}
	return 0, errors.Errorf("Key '%s' not found", key)
}

func GetFloat64(data map[string]any, key string) (float64, error) {
	if item, ok := data[key]; ok {
		if n, ok := item.(json.Number); ok {
			if v, err := strconv.ParseFloat(string(n), 64); err == nil {
				return v, nil
			}
			return 0, errors.Errorf("Key '%s' is not a number type", key)
		}
		return 0, errors.Errorf("Key '%s' is not a number type", key)
	}
	return 0, errors.Errorf("Key '%s' not found", key)
}

func GetInt8(data map[string]any, key string) (int8, error) {
	if v, err := GetInt64(data, key); err != nil {
		return 0, err
	} else {
		return int8(v), nil
	}
}

func GetInt16(data map[string]any, key string) (int16, error) {
	if v, err := GetInt64(data, key); err != nil {
		return 0, err
	} else {
		return int16(v), nil
	}
}

func GetInt32(data map[string]any, key string) (int32, error) {
	if v, err := GetInt64(data, key); err != nil {
		return 0, err
	} else {
		return int32(v), nil
	}
}

func GetInt(data map[string]any, key string) (int, error) {
	if v, err := GetInt64(data, key); err != nil {
		return 0, err
	} else {
		return int(v), nil
	}
}

func GetUint8(data map[string]any, key string) (uint8, error) {
	if v, err := GetUint64(data, key); err != nil {
		return 0, err
	} else {
		return uint8(v), nil
	}
}

func GetUint16(data map[string]any, key string) (uint16, error) {
	if v, err := GetUint64(data, key); err != nil {
		return 0, err
	} else {
		return uint16(v), nil
	}
}

func GetUint32(data map[string]any, key string) (uint32, error) {
	if v, err := GetUint64(data, key); err != nil {
		return 0, err
	} else {
		return uint32(v), nil
	}
}

func GetUint(data map[string]any, key string) (uint, error) {
	if v, err := GetUint64(data, key); err != nil {
		return 0, err
	} else {
		return uint(v), nil
	}
}

func GetFloat32(data map[string]any, key string) (float32, error) {
	if v, err := GetFloat64(data, key); err != nil {
		return 0, err
	} else {
		return float32(v), nil
	}
}

func GetBoolean(data map[string]any, key string) (bool, error) {
	if item, ok := data[key]; ok {
		if v, ok := item.(bool); ok {
			return v, nil
		}
		return false, errors.Errorf("Key '%s' is not a boolean type", key)
	}
	return false, errors.Errorf("Key '%s' not found", key)
}

func GetString(data map[string]any, key string) (string, error) {
	if item, ok := data[key]; ok {
		if s, ok := item.(string); ok {
			return s, nil
		}
		return "", errors.Errorf("Key '%s' is not a string type", key)
	}
	return "", errors.Errorf("Key '%s' not found", key)
}
