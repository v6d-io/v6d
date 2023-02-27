package util

import "github.com/spf13/cobra"

func AssertNoArgs(cmd *cobra.Command, args []string) {
	if err := cobra.NoArgs(cmd, args); err != nil {
		ErrLogger.Fatalf("Expects no positional arguments: %+v", err)
	}
}
