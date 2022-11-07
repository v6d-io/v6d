.. list-table:: Dask Reparition Drivers Configurations
   :widths: 25 15 60
   :header-rows: 1

   * - Name
     - Yaml Fields
     - Description

   * - "scheduling.k8s.v6d.io/dask-scheduler"
     - annotations
     - The service of dask scheduler.

   * - "scheduling.k8s.v6d.io/dask-worker-selector"
     - annotations
     - The label selector of dask worker pod.

   * - "repartition.v6d.io/enabled"
     - labels
     - Enable the repartition.

   * - "repartition.v6d.io/type"
     - labels
     - The type of repartition, at present, 
       only support `dask`. 

   * - "scheduling.k8s.v6d.io/replicas"
     - labels
     - The replicas of the workload.

   * - metadata
     - string
     - The same as typename
     - nil

   * - metadata
     - string
     - The same as typename
     - nil

   * - config.etcdCmd
     - string
     - The path of etcd command.
     - nil

   * - config.etcdEndpoint
     - string
     - The endpoint of etcd.
     - nil

   * - config.etcdPrefix
     - string
     - The path prefix of etcd.
     - nil

   * - config.enableMetrics
     - bool
     - Enable the metrics in vineyardd.
     - false

   * - config.enablePrometheus
     - bool
     - Enable the Prometheus.
     - false

   * - config.socket
     - string
     - The ipc socket file of vineyardd.
     - nil

   * - config.streamThreshold
     - int64
     - The memory threshold of streams (percentage of total memory) 
     - nil

   * - config.sharedMemorySize
     - string
     - The shared memory size for vineyardd.
     - nil

   * - config.syncCRDs
     - bool
     - Synchronize CRDs when persisting objects
     - true

   * - config.spillConfig
       .Name
     - string
     - The name of the spill config, if set we'll enable the spill module.
     - nil

   * - config.spillConfig
       .path
     - string
     - The path of spilling.
     - nil

   * - config.spillConfig
       .spillLowerRate
     - string
     - The low watermark of spilling memory.
     - nil

   * - config.spillConfig
       .spillUpperRate
     - string
     - The high watermark of triggering spilling.
     - nil

   * - config.spillConfig
       .persistentVolumeSpec
     - corev1
       .PersistentVolumeSpec
     - The PV of the spilling for persistent storage.
     - nil

   * - config.spillConfig
       .persistentVolumeClaimSpec
     - corev1.
       PersistentVolumeClaimSpec
     - The PVC of the spilling for the persistent storage.
     - nil

   * - service.type
     - string
     - The service type of vineyardd service.
     - nil

   * - service.port
     - int
     - The service port of vineyardd service 
     - nil

   * - etcd.replicas
     - int
     - The etcd replicas of vineyard
     - nil