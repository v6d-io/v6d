#!/usr/bin/env sh


wait_for() {
    echo Waiting for $1 to listen on $2...
    while ! nc -z $1 $2; do echo waiting...; sleep 1s; done
}

start_hdfs_namenode() {
  
	if [ ! -f /tmp/namenode-formated ];then
		${HADOOP_HOME}/bin/hdfs namenode -format >/tmp/namenode-formated 
	fi

	${HADOOP_HOME}/bin/hdfs --loglevel INFO namenode
	
	tail -f ${HADOOP_HOME}/logs/*namenode*.log
}

start_hdfs_datanode() {

        wait_for $1 $2
	
	${HADOOP_HOME}/bin/hdfs --loglevel INFO --daemon start datanode

    tail -f ${HADOOP_HOME}/logs/*datanode*.log	
}

start_yarn_resourcemanager() {

        ${HADOOP_HOME}/bin/yarn --loglevel INFO --daemon start resourcemanager

        tail -f ${HADOOP_HOME}/logs/*resourcemanager*.log
}

start_yarn_nodemanager() {

        wait_for $1 $2

        ${HADOOP_HOME}/bin/yarn --loglevel INFO --daemon start nodemanager

        tail -f ${HADOOP_HOME}/logs/*nodemanager*.log
}

start_yarn_proxyserver() {

        wait_for $1 $2

        ${HADOOP_HOME}/bin/yarn --loglevel INFO --daemon start proxyserver
        tail -f ${HADOOP_HOME}/logs/*proxyserver*.log
}

start_mr_historyserver() {
       
        wait_for $1 $2

	${HADOOP_HOME}/bin/mapred --loglevel INFO  --daemon  start historyserver

	tail -f ${HADOOP_HOME}/logs/*historyserver*.log
}

start_hive_metastore() {

	if [ ! -f ${HIVE_HOME}/formated ];then
		schematool -initSchema -dbType mysql --verbose >  ${HIVE_HOME}/formated
	fi
	$HIVE_HOME/bin/hive --service metastore

}

start_hive_hiveserver2() {

	$HIVE_HOME/bin/hive --service hiveserver2
}


case $1 in
	hadoop-hdfs-nn)
		start_hdfs_namenode
		;;
	hadoop-hdfs-dn)
		start_hdfs_datanode $2 $3
		;;
	hadoop-yarn-rm)
		start_yarn_resourcemanager
		;;
	hadoop-yarn-nm)
                start_yarn_nodemanager $2 $3
                ;;
	hadoop-yarn-proxyserver)
		start_yarn_proxyserver $2 $3
		;;
	hadoop-mr-historyserver)
		start_mr_historyserver $2 $3
		;;
	hive-metastore)
		start_hive_metastore $2 $3
		;;
	hive-hiveserver2)
		start_hive_hiveserver2 $2 $3
		;;
	*)
		echo "请输入正确的服务启动命令~"
	;;
esac

