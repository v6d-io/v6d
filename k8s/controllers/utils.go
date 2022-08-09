package controllers

import (
	"crypto/rand"
	"fmt"
)

func GenerateRandomName(length int) string {
	bs := make([]byte, length)
	if _, err := rand.Read(bs); err != nil {
		fmt.Println("rand.Read false: ", err)
	}
	return fmt.Sprintf("%x", bs)[:length]
}
