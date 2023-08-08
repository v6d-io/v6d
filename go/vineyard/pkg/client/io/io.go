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

package io

import (
	"encoding/binary"
	"net"
	"strconv"
	"time"

	"github.com/pkg/errors"
	"github.com/v6d-io/v6d/go/vineyard/pkg/common/log"
)

const (
	kNumConnectAttempts = 10
	kConnectTimeoutMs   = 1000
)

func ConnectIPCSocketRetry(pathname string, conn **net.UnixConn) error {
	var numRetries int = kNumConnectAttempts
	var timeout int64 = kConnectTimeoutMs

	err := ConnectIPCSocket(pathname, conn)
	for {
		if err == nil || numRetries < 0 {
			break
		}
		log.Infof(
			"Connecting to IPC socket failed for pathname %s with error %s, retrying %d more times.",
			pathname,
			err,
			numRetries,
		)
		time.Sleep(time.Duration(timeout) * time.Millisecond)
		err = ConnectIPCSocket(pathname, conn)
		numRetries--
	}
	return err
}

func ConnectIPCSocket(pathname string, conn **net.UnixConn) error {
	addr, err := net.ResolveUnixAddr("unix", pathname)
	if err != nil {
		return err
	}
	*conn, err = net.DialUnix("unix", nil, addr)
	if err != nil {
		return err
	}
	return nil
}

func ConnectRPCSocket(host string, port uint16, conn *net.Conn) error {
	tcpAddr, err := net.ResolveTCPAddr("tcp", host+":"+strconv.Itoa(int(port)))
	if err != nil {
		return err
	}
	*conn, err = net.DialTCP("tcp", nil, tcpAddr)
	if err != nil {
		return err
	}
	return nil
}

func ConnectRPCSocketRetry(host string, port uint16, conn *net.Conn) error {
	var numRetries int = kNumConnectAttempts
	var timeout int64 = kConnectTimeoutMs

	err := ConnectRPCSocket(host, port, conn)
	for {
		if err == nil || numRetries < 0 {
			break
		}
		log.Infof(
			"Connecting to RPC socket failed for endpoint %s:%d with error %s, retrying %d more times.",
			host,
			port,
			err,
			numRetries,
		)
		time.Sleep(time.Duration(timeout) * time.Millisecond)
		err = ConnectRPCSocket(host, port, conn)
		numRetries--
	}
	return err
}

func sendBytes(conn net.Conn, data []byte, length int) error {
	var bytesLeft int = length
	var offset int = 0
	for bytesLeft > 0 {
		nBytes, err := conn.Write(data[offset:length])
		if err != nil {
			return errors.Wrapf(err, "send message failed")
		}
		bytesLeft -= nBytes
		offset += nBytes
	}
	return nil
}

func recvBytes(conn net.Conn, data []byte, length int) error {
	var bytesLeft int = length
	var offset int = 0
	for bytesLeft > 0 {
		nBytes, err := conn.Read(data[offset:length])
		if err != nil {
			return errors.Wrapf(err, "receive message failed")
		}
		bytesLeft -= nBytes
		offset += nBytes
	}
	return nil
}

func SendMessageBytes(conn net.Conn, message []byte) error {
	length := len(message)
	lengthBytes := make([]byte, strconv.IntSize/8)
	binary.LittleEndian.PutUint64(lengthBytes, uint64(length))
	if err := sendBytes(conn, lengthBytes, strconv.IntSize/8); err != nil {
		return err
	}
	return sendBytes(conn, message, length)
}

func SendMessageString(conn net.Conn, message string) error {
	return SendMessageBytes(conn, []byte(message))
}

func RecvMessageBytes(conn net.Conn) (bytes []byte, err error) {
	lengthBytes := make([]byte, strconv.IntSize/8)

	if err := recvBytes(conn, lengthBytes, strconv.IntSize/8); err != nil {
		return nil, err
	}
	var length uint64 = binary.LittleEndian.Uint64(lengthBytes)
	stringBytes := make([]byte, length)
	if err := recvBytes(conn, stringBytes, int(length)); err != nil {
		return nil, err
	}
	return stringBytes, nil
}

func RecvMessageString(conn net.Conn) (string, error) {
	bytes, err := RecvMessageBytes(conn)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}
