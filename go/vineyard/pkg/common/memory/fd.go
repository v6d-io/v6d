package memory

import (
	"context"
	"syscall"
	"unsafe"

	"github.com/pkg/errors"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common/log"
)

func SendFileDescriptor(conn int, fd int) error {
	return errors.Errorf("Not implemented yet")
}

func RecvFileDescriptor(conn int) (int, error) {
	logger := log.FromContext(context.TODO())

	var oobn int
	var err error
	oob := make([]byte, syscall.CmsgSpace(int(unsafe.Sizeof(int32(0)))))
	for {
		_, oobn, _, _, err = syscall.Recvmsg(conn, nil, oob, 0)
		if err != nil {
			if err == syscall.EAGAIN || err == syscall.EWOULDBLOCK || err == syscall.EINTR {
				continue
			} else {
				logger.Error(err, "Error in recv_fd")
				return 0, errors.Wrapf(err, "Error in recv_fd")
			}
		} else {
			break
		}
	}
	messages, err := syscall.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		return 0, err
	}
	for _, scm := range messages {
		fds, err := syscall.ParseUnixRights(&scm)
		if err != nil {
			continue
		}
		if len(fds) > 0 {
			return fds[0], nil
		}
	}
	return 0, errors.Errorf("Failed to recv fd from remote server")
}
