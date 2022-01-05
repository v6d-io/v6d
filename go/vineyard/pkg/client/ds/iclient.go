package ds

type IIPCClient interface {
	CreateBlob(size int, blob *BlobWriter)
}
