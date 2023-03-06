package usage

import (
	"encoding/json"

	"github.com/spf13/cobra"
)

func UsageJson(cmd *cobra.Command) error {
	usage, err := CobraUsage(cmd)
	if err != nil {
		return err
	}
	// note that `goccy/go-json` won't here (hung up) for `MarshalIndent`,
	usageJson, err := json.MarshalIndent(usage, "", "    ")
	if err != nil {
		return err
	}
	cmd.Printf("%+v", string(usageJson))
	return nil
}
