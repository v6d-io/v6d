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
	
	tail -f /var/log/hadoop/*namenode*.log
}

start_hdfs_datanode() {
    wait_for $1 $2
	
	${HADOOP_HOME}/bin/hdfs --loglevel INFO --daemon start datanode
    tail -f /var/log/hadoop/*datanode*.log	
}

start_yarn_resourcemanager() {
    ${HADOOP_HOME}/bin/yarn --loglevel INFO --daemon start resourcemanager
    tail -f /var/log/hadoop/*resourcemanager*.log
}

start_yarn_nodemanager() {
	wait_for $1 $2

	${HADOOP_HOME}/bin/yarn --loglevel INFO --daemon start nodemanager
	tail -f /var/log/hadoop/*nodemanager*.log
}

start_yarn_proxyserver() {
	wait_for $1 $2

	${HADOOP_HOME}/bin/yarn --loglevel INFO --daemon start proxyserver
	tail -f /var/log/hadoop/*proxyserver*.log
}

start_mr_historyserver() {
       
    wait_for $1 $2

	${HADOOP_HOME}/bin/mapred --loglevel INFO  --daemon  start historyserver
	tail -f /var/log/hadoop/*historyserver*.log
}

start_hive_metastore() {
	DB_DRIVER=derby
	if [ "$1" = "local" ]; then
		cp /hive-config/* $HIVE_HOME/conf/
	else
		cp /hive-config-distributed/* $HIVE_HOME/conf/
		DB_DRIVER=mysql
	fi

	if [ ! -f ${HIVE_HOME}/formated ];then
		schematool -initSchema -dbType $DB_DRIVER --verbose >  ${HIVE_HOME}/formated
	fi
	$HIVE_HOME/bin/hive --skiphbasecp --service metastore

}

start_hive_hiveserver2() {
	if [ "$1" = "local" ]; then
		cp /hive-config/* $HIVE_HOME/conf/
	else
		cp /hive-config-distributed/* $HIVE_HOME/conf/
	fi

	$HIVE_HOME/bin/hive --skiphbasecp --service hiveserver2
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
		start_hive_metastore $2 $3 $4
		;;
	hive-hiveserver2)
		start_hive_hiveserver2 $2 $3 $4
		;;
	*)
		echo "请输入正确的服务启动命令~"
	;;
esac

