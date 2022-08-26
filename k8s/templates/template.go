package templates

import (
	"embed"
	"fmt"
	"path/filepath"
)

//go:embed vineyardd
var F embed.FS

// EmbedTemplate is only used for implementing the interface
type EmbedTemplate struct{}

func NewEmbedTemplate() *EmbedTemplate {
	return &EmbedTemplate{}
}
func (e *EmbedTemplate) ReadFile(path string) ([]byte, error) {
	return F.ReadFile(path)
}
func (e *EmbedTemplate) GetFilesRecursive(dir string) ([]string, error) {
	path := filepath.Join(filepath.Dir(dir), dir)
	fd, err := F.ReadDir(path)
	if err != nil {
		return []string{}, fmt.Errorf("ReadDir Error: %v", err)
	}
	files := []string{}
	for _, f := range fd {
		if !f.IsDir() {
			files = append(files, filepath.Join(path, f.Name()))
		}
	}
	return files, nil
}
