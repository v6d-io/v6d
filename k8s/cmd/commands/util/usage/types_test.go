package usage

import (
	"testing"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

type stringValue struct {
	value string
}

func newStringValue(val string) *stringValue {
	return &stringValue{value: val}
}

func (s *stringValue) Set(val string) error {
	s.value = val
	return nil
}

func (s *stringValue) Type() string {
	return "string"
}

func (s *stringValue) String() string {
	return s.value
}

func TestFlagUsage(t *testing.T) {
	// 创建一个 pflag.Flag 对象
	pflag := &pflag.Flag{
		Name:       "testFlag",
		Shorthand:  "t",
		Value:      newStringValue("test"),
		DefValue:   "test",
		Usage:      "This is a test flag",
		Deprecated: "This flag is deprecated",
	}

	// 调用 FlagUsage 函数
	flag := FlagUsage(pflag)

	// 检查返回的 Flag 对象的各个属性是否与 pflag.Flag 对象的属性一致
	if flag.Name != pflag.Name {
		t.Errorf("Expected Name to be %s, but got %s", pflag.Name, flag.Name)
	}

	if flag.Shorthand != pflag.Shorthand {
		t.Errorf("Expected Shorthand to be %s, but got %s", pflag.Shorthand, flag.Shorthand)
	}

	if flag.Type != pflag.Value.Type() {
		t.Errorf("Expected Type to be %s, but got %s", pflag.Value.Type(), flag.Type)
	}

	if flag.Default != pflag.DefValue {
		t.Errorf("Expected Default to be %s, but got %s", pflag.DefValue, flag.Default)
	}

	if flag.Help != pflag.Usage {
		t.Errorf("Expected Help to be %s, but got %s", pflag.Usage, flag.Help)
	}

	if flag.Deprecated != pflag.Deprecated {
		t.Errorf("Expected Deprecated to be %s, but got %s", pflag.Deprecated, flag.Deprecated)
	}
}

func TestCobraUsage(t *testing.T) {
	// 创建主命令和子命令
	mainCmd := &cobra.Command{
		Use:     "mainCmd",
		Short:   "Main command",
		Long:    "This is the main command",
		Example: "mainCmd --option1",
	}
	childCmd1 := &cobra.Command{
		Use:     "childCmd1",
		Short:   "Child command 1",
		Long:    "This is the first child command",
		Example: "mainCmd childCmd1 --option1",
	}
	childCmd2 := &cobra.Command{
		Use:     "childCmd2",
		Short:   "Child command 2",
		Long:    "This is the second child command",
		Example: "mainCmd childCmd2 --option1",
	}

	// 添加子命令到主命令
	mainCmd.AddCommand(childCmd1, childCmd2)

	// 添加标记和全局标记
	mainCmd.Flags().Bool("option1", false, "Option 1 for mainCmd")
	mainCmd.PersistentFlags().String("globalOption1", "", "Global option 1 for mainCmd")
	childCmd1.Flags().Bool("option1", false, "Option 1 for childCmd1")

	// 获取主命令的用法
	mainUsage, err := CobraUsage(mainCmd)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// 检查返回的Command对象的各个属性是否与主命令相匹配
	if mainUsage.Name != mainCmd.Name() {
		t.Errorf("Expected Name to be %s, but got %s", mainCmd.Name(), mainUsage.Name)
	}
	if mainUsage.Brief != mainCmd.Short {
		t.Errorf("Expected Brief to be %s, but got %s", mainCmd.Short, mainUsage.Brief)
	}
	if mainUsage.Description != mainCmd.Long {
		t.Errorf("Expected Description to be %s, but got %s", mainCmd.Long, mainUsage.Description)
	}
	if mainUsage.Usage != mainCmd.UseLine() {
		t.Errorf("Expected Usage to be %s, but got %s", mainCmd.UseLine(), mainUsage.Usage)
	}
	if mainUsage.Example != mainCmd.Example {
		t.Errorf("Expected Example to be %s, but got %s", mainCmd.Example, mainUsage.Example)
	}
	if mainUsage.Deprecated != mainCmd.Deprecated {
		t.Errorf("Expected Deprecated to be %s, but got %s", mainCmd.Deprecated, mainUsage.Deprecated)
	}
	if mainUsage.Runnable != mainCmd.Runnable() {
		t.Errorf("Expected Runnable to be %v, but got %v", mainCmd.Runnable(), mainUsage.Runnable)
	}

	// 检查返回的子命令列表的长度是否正确
	if len(mainUsage.Children) != 2 {
		t.Errorf("Expected 2 children, but got %d", len(mainUsage.Children))
	}

	// 检查返回的第一个子命令的所有属性是否与实际的子命令相匹配
	firstChildUsage := mainUsage.Children[0]
	if firstChildUsage.Name != childCmd1.Name() {
		t.Errorf("Expected Name of the first child to be %s, but got %s", childCmd1.Name(), firstChildUsage.Name)
	}

	// 检查标记和全局标记的数量
	if len(mainUsage.Flags) != 2 {
		t.Errorf("Expected 2 flag, but got %d", len(mainUsage.Flags))
	}
	if len(mainUsage.GlobalFlags) != 0 {
		t.Errorf("Expected 0 global flag, but got %d", len(mainUsage.GlobalFlags))
	}
}
