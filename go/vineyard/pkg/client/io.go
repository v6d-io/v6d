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

package vineyard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"strconv"
	"time"
)

const kNumConnectAttempts = 10
const kConnectTimeoutMs = 1000

func ConnectIPCSocketRetry(pathname string, conn **net.UnixConn) error {
	var numRetries int = kNumConnectAttempts
	var timeout int64 = kConnectTimeoutMs

	err := ConnectIPCSocket(pathname, conn)
	for {
		if err == nil || numRetries < 0 {
			break
		}
		fmt.Println("Connecting to IPC socket failed for pathname", pathname, "with ret err", err.Error(),
			", retrying", numRetries, "more times.")
		time.Sleep(time.Duration(timeout) * time.Millisecond)
		err = ConnectIPCSocket(pathname, conn)
		numRetries--
	}
	if err != nil {
		return err
	}
	return nil
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
	var err error
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
		fmt.Println("Connecting to RPC socket failed for endpoint", host, ":", port, "with ret = ", err.Error(),
			", retrying", numRetries, "more times.")
		time.Sleep(time.Duration(timeout) * time.Millisecond)
		err = ConnectRPCSocket(host, port, conn)
		numRetries--
	}
	if err != nil {
		return err
	}
	return nil
}

func SendBytes(conn net.Conn, data []byte, length int) error {
	var bytesLeft int = length
	var offset int = 0
	for bytesLeft > 0 {
		nBytes, err := conn.Write(data[offset:length])
		if err != nil {
			return errors.New(fmt.Sprintf("Send message failed :%s", err.Error()))
		}
		bytesLeft -= nBytes
		offset += nBytes
	}
	return nil
}

func SendMessage(conn net.Conn, msg string) error {
	// TODO: check 32 bits platform is also works
	length := len(msg)
	lengthBytes := make([]byte, strconv.IntSize/8)
	binary.LittleEndian.PutUint64(lengthBytes, uint64(length))
	if err := SendBytes(conn, lengthBytes, strconv.IntSize/8); err != nil {
		return err
	}
	if err := SendBytes(conn, []byte(msg), length); err != nil {
		return err
	}
	return nil
}

func RecvBytes(conn net.Conn, data []byte, length int) error {
	var bytesLeft int = length
	var offset int = 0
	for bytesLeft > 0 {
		nBytes, err := conn.Read(data[offset:length])
		if err != nil {
			return errors.New(fmt.Sprintf("Receive message failed :%s", err.Error()))
		}
		bytesLeft -= nBytes
		offset += nBytes
	}
	return nil
}

func RecvMessage(conn net.Conn, msg *string) error {
	lengthBytes := make([]byte, strconv.IntSize/8)

	if err := RecvBytes(conn, lengthBytes, strconv.IntSize/8); err != nil {
		return err
	}
	var length uint64 = binary.LittleEndian.Uint64(lengthBytes)
	stringBytes := make([]byte, length)
	if err := RecvBytes(conn, stringBytes, int(length)); err != nil {
		return err
	}
	*msg = string(stringBytes)
	return nil
}
