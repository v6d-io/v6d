package controllers

import (
	"crypto/rand"
	"fmt"
)

func GenerateRandomName(length int) string {
	bs := make([]byte, length)
	rand.Read(bs)
	return fmt.Sprintf("%x", bs)[:length]
}
