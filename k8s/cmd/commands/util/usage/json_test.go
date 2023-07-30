package usage

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/spf13/cobra"
)

func TestUsageJson(t *testing.T) {
	// 创建一个命令
	cmd := &cobra.Command{
		Use:   "root",
		Short: "Root command",
		Long:  "This is the root command",
		Run: func(cmd *cobra.Command, args []string) {
			_ = UsageJson(cmd)
		},
	}

	// 调用 UsageJson 函数并捕获输出
	var buf bytes.Buffer
	cmd.SetOut(&buf)
	err := UsageJson(cmd)
	if err != nil {
		t.Fatalf("Error generating JSON usage: %v", err)
	}

	// 解析输出的 JSON
	var usageMap map[string]interface{}
	err = json.Unmarshal(buf.Bytes(), &usageMap)
	if err != nil {
		t.Fatalf("Error parsing JSON usage: %v", err)
	}

	// 验证解析后的 JSON 结构是否符合预期
	expectedUsageMap := map[string]interface{}{
		"Brief":       "Root command",
		"Children":    []interface{}{},
		"Deprecated":  "",
		"Description": "This is the root command",
		"Example":     "",
		"Flags":       []interface{}{},
		"GlobalFlags": []interface{}{},
		"Name":        "root",
		"Runnable":    true,
		"Usage":       "root",
	}
	if !reflect.DeepEqual(usageMap, expectedUsageMap) {
		t.Errorf("Expected usage map %v, but got %v", expectedUsageMap, usageMap)
	}
}
