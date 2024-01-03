/*
* Copyright 2020-2023 Alibaba Group Holding Limited.

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
package util

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"os"
	"path"
	"reflect"
	"time"

	"github.com/pkg/errors"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/client-go/util/cert"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var (
	CACommonName   = "vineyard-webhook-ca"
	CAOrgainzation = []string{"Alibaba"}

	// CAKeyName is the name of the CA private key
	CAKeyName = "ca-key.pem"
	// CACertName is the name of the CA certificate
	CACertName = "ca-cert.pem"
	// WebhookServerKeyName is the name of the webhook server private key
	WebhookServerKeyName = "key.pem"
	// WebhookServerKeyForTLSName is the name of the webhook server private key for TLS
	WebhookServerKeyForTLSName = "tls.key"
	// WebhookServerCertName is the name of the webhook server certificate
	WebhookServerCertName = "cert.pem"
	// WebhookServerCertForTLSName is the name of the webhook server certificate for TLS
	WebhookServerCertForTLSName = "tls.crt"
)

// CertificateAuthority represents a self-signed CA
type CertificateAuthority struct {
	Cert *x509.Certificate
	Key  *rsa.PrivateKey
}

// WebhookCertificate contains the certificate and key of webhook server
type WebhookCertificate struct {
	Cert *x509.Certificate
	Key  *rsa.PrivateKey
}

type CertGenerator struct {
	// CommonName is the common name of the certificate
	CommonName string
	// Organizations is the organization of the certificate
	Organizations []string
	// DNSName is the DNS name of the certificate
	DNSName string
	// Directory is the directory to store the generated certificates
	Directory string
	CertificateAuthority
	WebhookCertificate
}

func NewCertGenerator(commonName string, organizations []string, dnsName string, dir string) (*CertGenerator, error) {
	ca, err := NewCertificateAuthority(commonName, organizations)
	if err != nil {
		return nil, errors.Wrap(err, "unable to generate CA")
	}

	wbcert, err := NewWebhookCertificate(commonName, organizations, ca, dnsName)
	if err != nil {
		return nil, errors.Wrap(err, "unable to generate webhook certificate")
	}
	return &CertGenerator{
		CommonName:    commonName,
		Organizations: organizations,
		DNSName:       dnsName,
		Directory:     dir,
		CertificateAuthority: CertificateAuthority{
			Cert: ca.Cert,
			Key:  ca.Key,
		},
		WebhookCertificate: WebhookCertificate{
			Cert: wbcert.Cert,
			Key:  wbcert.Key,
		},
	}, nil
}

func (c *CertGenerator) Generate() error {
	if err := WriteCertstoDir(c.Directory, &c.CertificateAuthority, &c.WebhookCertificate); err != nil {
		return errors.Wrap(err, "unable to write certificates to directory")
	}

	return nil
}

func (c *CertGenerator) getCABundle() ([]byte, error) {
	return pemEncodeToBytes("CERTIFICATE", c.CertificateAuthority.Cert)
}

func (c *CertGenerator) PatchCABundleToMutatingWebhook(webhookName string) error {
	cli := KubernetesClient()
	pemData, err := c.getCABundle()
	if err != nil {
		return err
	}

	mutatingWebhook := v1.MutatingWebhookConfiguration{}
	ctx := context.Background()
	if err := cli.Get(ctx, client.ObjectKey{Name: webhookName}, &mutatingWebhook); err != nil {
		return err
	}
	currentMutatingWebhook := mutatingWebhook.DeepCopy()
	for i := range mutatingWebhook.Webhooks {
		mutatingWebhook.Webhooks[i].ClientConfig.CABundle = pemData
	}
	if reflect.DeepEqual(mutatingWebhook.Webhooks, currentMutatingWebhook.Webhooks) {
		return nil
	}
	if err := cli.Patch(ctx, &mutatingWebhook, client.MergeFrom(currentMutatingWebhook)); err != nil {
		return err
	}
	return nil
}
func (c *CertGenerator) PatchCABundleToValidatingWebhook(webhookName string) error {
	cli := KubernetesClient()
	pemData, err := c.getCABundle()
	if err != nil {
		return err
	}

	validatingWebhook := v1.ValidatingWebhookConfiguration{}
	ctx := context.Background()
	if err := cli.Get(ctx, client.ObjectKey{Name: webhookName}, &validatingWebhook); err != nil {
		return err
	}
	currentValidatingWebhook := validatingWebhook.DeepCopy()
	for i := range validatingWebhook.Webhooks {
		validatingWebhook.Webhooks[i].ClientConfig.CABundle = pemData
	}
	if reflect.DeepEqual(validatingWebhook.Webhooks, currentValidatingWebhook.Webhooks) {
		return nil
	}
	if err := cli.Patch(ctx, &validatingWebhook, client.MergeFrom(currentValidatingWebhook)); err != nil {
		return err
	}
	return nil
}

// NewCertificateAuthority create a new self-signed CA
func NewCertificateAuthority(commonName string, organizations []string) (*CertificateAuthority, error) {
	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("unable to generate CA private key: %w", err)
	}

	caConfig := cert.Config{
		CommonName:   commonName,
		Organization: organizations,
	}
	caCert, err := cert.NewSelfSignedCACert(caConfig, caKey)
	if err != nil {
		return nil, fmt.Errorf("unable to create self-signed CA cert: %w", err)
	}

	return &CertificateAuthority{
		Cert: caCert,
		Key:  caKey,
	}, nil
}

// NewWebhookCertificate create a new certificate for webhook server
func NewWebhookCertificate(commonName string, organizations []string, ca *CertificateAuthority, dnsName string) (*WebhookCertificate, error) {
	if err := validateCACertificate(ca.Cert); err != nil {
		fmt.Println("ca cert is invalid, generating a new one...")
		if ca, err = NewCertificateAuthority(commonName, organizations); err != nil {
			return nil, errors.Wrap(err, "unable to generate CA")
		}
	}

	serverKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("unable to generate server private key: %w", err)
	}

	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("unable to generate serial number: %w", err)
	}

	if err = WaitForDNSName(dnsName, 60*time.Second); err != nil {
		return nil, fmt.Errorf("unable to resolve DNS name: %w", err)
	}
	ipAddresses, err := net.LookupIP(dnsName)
	if err != nil {
		return nil, fmt.Errorf("unable to lookup IP address: %w", err)
	}
	dnsNames := []string{dnsName, "localhost"}

	serverCertTemplate := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:   commonName,
			Organization: ca.Cert.Subject.Organization,
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(365 * 24 * time.Hour), // 1 year

		KeyUsage:    x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},

		BasicConstraintsValid: true,
		DNSNames:              dnsNames,
		IPAddresses:           ipAddresses,
	}

	serverCertDER, err := x509.CreateCertificate(rand.Reader, &serverCertTemplate, ca.Cert, &serverKey.PublicKey, ca.Key)
	if err != nil {
		return nil, fmt.Errorf("unable to create server cert: %w", err)
	}

	serverCert, err := x509.ParseCertificate(serverCertDER)
	if err != nil {
		return nil, fmt.Errorf("unable to parse server certificate: %w", err)
	}

	return &WebhookCertificate{
		Cert: serverCert,
		Key:  serverKey,
	}, nil
}

// validateCACertificate validates the given CA certificate
func validateCACertificate(caCert *x509.Certificate) error {
	// check if the certificate is a CA
	if !caCert.IsCA {
		return errors.New("certificate is not a CA")
	}

	// check if the certificate is self-signed
	if caCert.CheckSignatureFrom(caCert) != nil {
		return errors.New("certificate is not self-signed")
	}

	// check if the certificate is valid
	now := time.Now()
	if now.Before(caCert.NotBefore) {
		return fmt.Errorf("certificate is not valid yet (valid from %s)", caCert.NotBefore)
	}
	if now.After(caCert.NotAfter) {
		return fmt.Errorf("certificate has expired (expired on %s)", caCert.NotAfter)
	}

	return nil
}

func pemEncodeToBytes(pemType string, data interface{}) ([]byte, error) {
	var pemBlock *pem.Block

	switch key := data.(type) {
	case *rsa.PrivateKey:
		derFormat := x509.MarshalPKCS1PrivateKey(key)
		pemBlock = &pem.Block{Type: pemType, Bytes: derFormat}

	case *x509.Certificate:
		pemBlock = &pem.Block{Type: pemType, Bytes: key.Raw}

	default:
		return nil, errors.New("unsupported data type for PEM encoding")
	}

	return pem.EncodeToMemory(pemBlock), nil
}

// writeCertsToFile writes the pem-encoded data to the given file
func writeCertsToFile(filename, pemType string, data interface{}) error {
	pemData, err := pemEncodeToBytes(pemType, data)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filename, pemData, 0600); err != nil {
		return err
	}
	return nil
}

// WriteCertstoDir writes the given data to the given file under the given directory
func WriteCertstoDir(dir string, ca *CertificateAuthority, wbcert *WebhookCertificate) error {
	if err := os.MkdirAll(dir, 0600); err != nil {
		return err
	}

	caKey := path.Join(dir, CAKeyName)
	if err := writeCertsToFile(caKey, "RSA PRIVATE KEY", ca.Key); err != nil {
		return err
	}

	caCert := path.Join(dir, CACertName)
	if err := writeCertsToFile(caCert, "CERTIFICATE", ca.Cert); err != nil {
		return err
	}

	serverKey := path.Join(dir, WebhookServerKeyName)
	if err := writeCertsToFile(serverKey, "RSA PRIVATE KEY", wbcert.Key); err != nil {
		return err
	}

	serverKeyForTLS := path.Join(dir, WebhookServerKeyForTLSName)
	if err := writeCertsToFile(serverKeyForTLS, "RSA PRIVATE KEY", wbcert.Key); err != nil {
		return err
	}

	serverCert := path.Join(dir, WebhookServerCertName)
	if err := writeCertsToFile(serverCert, "CERTIFICATE", wbcert.Cert); err != nil {
		return err
	}

	serverCertForTLS := path.Join(dir, WebhookServerCertForTLSName)
	if err := writeCertsToFile(serverCertForTLS, "CERTIFICATE", wbcert.Cert); err != nil {
		return err
	}

	return nil
}

// wait for a DNS name to be resolved with a timeout
func WaitForDNSName(dnsName string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for DNS name %s to be resolved", dnsName)
		default:
			_, err := net.LookupIP(dnsName)
			if err == nil {
				return nil
			}
			time.Sleep(1 * time.Second)
		}
	}
}
