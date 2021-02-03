apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: vineyard
  namespace: vineyard
  labels:
    app: vineyard
spec:
  selector:
    matchLabels:
      name: vineyard
  template:
    metadata:
      labels:
        name: vineyard
    spec:
      tolerations:
      # this toleration is to have the daemonset runnable on master nodes
      # remove it if your masters can't run pods
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      containers:
      - name: vineyard
        image: quay.io/libvineyard/vineyardd:latest
        args:
        - "--socket"
        - "{Socket}"
        - "--etcd_endpoint"
        - "http://etcd-for-vineyard.{Namespace}.svc.cluster.local:2379"
        - "--size"
        - "{Size}"
        resources:
          limits:
            memory: 200m
          requests:
            cpu: 100m
            memory: 200m
        volumeMounts:
        - name: varrun
          mountPath: /var/run
        - name: shm
          mountPath: /dev/shm
      terminationGracePeriodSeconds: 30
      volumes:
      - name: varrun
        hostPath:
          path: /var/run/vineyard
      - name: shm
        emptyDir:
          medium: Memory


---
apiVersion: v1
kind: Service
metadata:
  name: vineyard-rpc
  namespace: vineyard
spec:
  selector:
    app: vineyard
  ports:
    - protocol: TCP
      port: {Port}
      targetPort: {Port}
