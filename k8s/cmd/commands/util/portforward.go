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
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
)

const (
	// ForwardAddress is the address to forward to
	ForwardAddress = "127.0.0.1"

	// DeploymentLabel is the label used to select the pod under a deployment
	DeploymentLabel = "app.vineyard.io/name"
)

func PortforwardDeployment(deploymentName, namespace string, forwardPort, port int, readyChannel, stopChannel chan struct{}) {
	log.Println("start")
	restConfig := GetKubernetesConfig()
	clientSet := KubernetesClientset()

	podList, err := clientSet.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: DeploymentLabel + "=" + deploymentName,
	})
	if err != nil {
		log.Fatalf("failed to get pods for deployment %s: %v", deploymentName, err)
	}
	if len(podList.Items) == 0 {
		log.Fatalf("no pods found for deployment %s in namespace %s", deploymentName, namespace)
	}

	podName := podList.Items[0].Name

	req := clientSet.CoreV1().RESTClient().Post().Namespace(namespace).
		Resource("pods").Name(podName).SubResource("portforward")
	log.Println(req.URL())
	signals := make(chan os.Signal, 1)

	defer signal.Stop(signals)

	signal.Notify(signals, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-signals
		close(stopChannel)
	}()

	if err := forwardPorts("POST", req.URL(), restConfig, stopChannel, readyChannel, forwardPort, port); err != nil {
		log.Fatal(err, "Fail to forward ports")
	}

}

func forwardPorts(method string, url *url.URL, config *rest.Config, stopChannel, readyChannel chan struct{}, forwardPort, port int) error {
	transport, upgrader, err := spdy.RoundTripperFor(config)
	if err != nil {
		return err
	}
	address := []string{ForwardAddress}
	ports := []string{strconv.Itoa(forwardPort) + ":" + strconv.Itoa(port)}

	IOStreams := struct {
		In     io.Reader
		Out    io.Writer
		ErrOut io.Writer
	}{os.Stdin, os.Stdout, os.Stderr}

	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, method, url)
	fw, err := portforward.NewOnAddresses(dialer, address, ports, stopChannel, readyChannel, IOStreams.Out, IOStreams.ErrOut)
	if err != nil {
		return err
	}
	return fw.ForwardPorts()
}
