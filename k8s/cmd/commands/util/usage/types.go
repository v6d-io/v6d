package usage

import (
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

type Flag struct {
	Name       string
	Shorthand  string
	Type       string
	Default    string
	Help       string
	Deprecated string
}

type Command struct {
	Name        string
	Brief       string
	Description string
	Usage       string
	Example     string
	Deprecated  string
	Runnable    bool
	Children    []Command
	Flags       []Flag
	GlobalFlags []Flag
}

func FlagUsage(flag *pflag.Flag) Flag {
	return Flag{
		Name:       flag.Name,
		Shorthand:  flag.Shorthand,
		Type:       flag.Value.Type(),
		Default:    flag.DefValue,
		Help:       flag.Usage,
		Deprecated: flag.Deprecated,
	}
}

func CobraUsage(cmd *cobra.Command) (Command, error) {
	c := Command{
		Name:        cmd.Name(),
		Brief:       cmd.Short,
		Description: cmd.Long,
		Usage:       cmd.UseLine(),
		Example:     cmd.Example,
		Deprecated:  cmd.Deprecated,
		Runnable:    cmd.Runnable(),
		Children:    []Command{},
		Flags:       []Flag{},
		GlobalFlags: []Flag{},
	}
	for _, child := range cmd.Commands() {
		childUsage, err := CobraUsage(child)
		if err != nil {
			return c, err
		}
		c.Children = append(c.Children, childUsage)
	}
	cmd.LocalFlags().VisitAll(func(flag *pflag.Flag) {
		c.Flags = append(c.Flags, FlagUsage(flag))
	})
	cmd.InheritedFlags().VisitAll(func(flag *pflag.Flag) {
		c.GlobalFlags = append(c.GlobalFlags, FlagUsage(flag))
	})
	return c, nil
}
